from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.table import Table

from src.config import get_settings
from src.ingest.clone import clone_repo, derive_slug, write_meta
from src.ingest.chunker import chunk_all, write_chunks
from src.ingest.git_pairs import extract_git_pairs, write_git_pairs
from src.ingest.callgraph import extract_call_graph, write_call_graph
from src.rag.store import build_index
from src.logging import get_logger, setup_logging

app = typer.Typer(
    name="repomate",
    help="repomate: Code If You Can — self-training repo-native coding copilot",
)

logger = get_logger(__name__)
console = Console()


@app.command()
def init(
    repo_url: str,
    force: bool = typer.Option(False, "--force", help="Force re-clone even if repo exists"),
) -> None:
    """Clone, chunk, and index a repo. LLM-free. RAG-ready with no training."""
    settings = get_settings()
    slug = derive_slug(repo_url)
    repo_dir = settings.data_dir / slug / "repo"

    for subdir in ("repo", "chroma", "adapters"):
        (settings.data_dir / slug / subdir).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    console.print(f"\n[bold]repomate init[/bold] — {repo_url}")
    console.print(f"Slug: [cyan]{slug}[/cyan]")

    console.print("\n[bold]Cloning...[/bold]")
    try:
        clone_repo(repo_url, repo_dir, force=force)
    except RuntimeError as e:
        console.print(f"[red]Clone failed:[/red] {e}")
        raise typer.Exit(code=1)

    from src.ingest.clone import get_head_sha
    head_sha = get_head_sha(repo_dir)
    sha8 = head_sha[:8]

    console.print("\n[bold]Chunking...[/bold]")
    chunks, chunk_stats = chunk_all(repo_dir, slug, sha8)
    write_chunks(slug, chunks)
    console.print(f"  {chunk_stats['total_files']} files → {chunk_stats['total_chunks']} chunks")

    console.print("\n[bold]Git pairs...[/bold]")
    clean, relabel, discarded = extract_git_pairs(repo_dir, slug)
    write_git_pairs(slug, clean, relabel)
    console.print(f"  clean={len(clean)}  to-relabel={len(relabel)}  discarded={discarded}")

    console.print("\n[bold]Call graph...[/bold]")
    graph = extract_call_graph(slug, repo_dir)
    write_call_graph(slug, graph)
    console.print(f"  {len(graph.nodes)} nodes  {len(graph.edges)} edges")

    console.print("\n[bold]Indexing...[/bold]")
    index_count = build_index(slug)
    console.print(f"  {index_count} vectors indexed")

    elapsed = time.perf_counter() - t0

    stats = {
        "files": chunk_stats["total_files"],
        "chunks": chunk_stats["total_chunks"],
        "by_language": chunk_stats["by_language"],
        "clean_pairs": len(clean),
        "relabel_pairs": len(relabel),
        "discarded_pairs": discarded,
        "call_graph_nodes": len(graph.nodes),
        "call_graph_edges": len(graph.edges),
        "elapsed_s": round(elapsed, 1),
    }

    write_meta(slug, repo_dir, stats)

    table = Table(title="Ingest Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files", str(stats["files"]))
    table.add_row("Chunks", str(stats["chunks"]))
    table.add_row("Call graph nodes", str(stats["call_graph_nodes"]))
    table.add_row("Call graph edges", str(stats["call_graph_edges"]))
    table.add_row("Clean git pairs", str(stats["clean_pairs"]))
    table.add_row("To relabel", str(stats["relabel_pairs"]))
    table.add_row("Discarded", str(stats["discarded_pairs"]))
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    console.print(table)


@app.command()
def index(repo_slug: str) -> None:
    """Re-embed changed chunks for an existing repo slug."""
    settings = get_settings()
    chunks_path = settings.data_dir / repo_slug / "chunks.jsonl"
    if not chunks_path.exists():
        console.print(f"[red]No chunks found for '{repo_slug}' — run `repomate init` first.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Indexing {repo_slug}...[/bold]")
    count = build_index(repo_slug)
    console.print(f"[green]Indexed {count} chunks.[/green]")


@app.command()
def train(
    repo_slug: str = typer.Argument(None),
    epochs: int = typer.Option(1, "--epochs", help="Number of training epochs"),
    lr: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(2, "--batch-size", help="Per-device batch size"),
    max_chunks: int = typer.Option(200, "--max-chunks", help="Max chunks for synth (cost control)"),
    skip_data: bool = typer.Option(False, "--skip-data", help="Skip data generation, only train"),
) -> None:
    """Generate instruction data + QLoRA fine-tune. Opt-in, periodic refresh."""
    import json as _json

    settings = get_settings()
    if not repo_slug:
        console.print("[red]Please specify a repo slug.[/red]")
        raise typer.Exit(code=1)

    data_dir = settings.data_dir / repo_slug
    if not data_dir.exists():
        console.print(f"[red]Repo '{repo_slug}' not found — run `repomate init` first.[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]repomate train — {repo_slug}[/bold]")

    if not skip_data:
        console.print("\n[bold]Phase 3: Data generation...[/bold]")
        chunks_path = data_dir / "chunks.jsonl"
        call_graph_path = data_dir / "call_graph.json"

        chunks = []
        with open(chunks_path) as f:
            for line in f:
                if line.strip():
                    chunks.append(_json.loads(line))

        call_graph = {}
        if call_graph_path.exists():
            call_graph = _json.loads(call_graph_path.read_text())

        from src.data.synth_pairs import generate_synth_pairs
        from src.data.contextual_pairs import generate_contextual_pairs
        from src.data.file_summary_pairs import generate_file_summary_pairs
        from src.data.impact_pairs import generate_impact_pairs
        from src.data.diff_relabel import generate_relabel_pairs
        from src.data.ast_pairs import generate_ast_pairs

        n = settings.synth_pairs_per_chunk
        console.print(f"  Generating synth pairs (max {max_chunks} chunks, {n}/chunk)...")
        generate_synth_pairs(chunks, data_dir / "synth_pairs.jsonl", n_per_chunk=n, max_chunks=max_chunks)

        console.print("  Generating contextual pairs...")
        generate_contextual_pairs(chunks, call_graph, data_dir / "contextual_pairs.jsonl", max_chunks=max_chunks)

        console.print("  Generating file-summary pairs...")
        generate_file_summary_pairs(chunks, data_dir / "file_summary_pairs.jsonl")

        console.print("  Generating impact pairs...")
        generate_impact_pairs(chunks, call_graph, data_dir / "impact_pairs.jsonl")

        console.print("  Generating diff-relabel pairs...")
        relabel_path = data_dir / "commits_to_relabel.jsonl"
        generate_relabel_pairs(relabel_path, data_dir / "relabel_pairs.jsonl")

        console.print("  Generating AST pairs (free, no LLM)...")
        generate_ast_pairs(chunks, call_graph, data_dir / "ast_pairs.jsonl")

        console.print("  Assembling train.jsonl + eval.jsonl + qa_questions.jsonl...")
        from src.data.assemble import assemble
        stats = assemble(repo_slug)
        console.print(f"  {stats['train_pairs']} train, {stats['eval_pairs']} eval, {stats['qa_questions']} QA questions")
        console.print(f"  Sources: {stats['by_source']}")

        from src.data.llm_client import get_cost_tracker
        get_cost_tracker().log()

    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        console.print(f"[red]No train.jsonl for '{repo_slug}'. Run without --skip-data.[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]Phase 5: QLoRA fine-tuning...[/bold]")
    console.print(f"  epochs={epochs}  lr={lr}  batch_size={batch_size}")

    from src.train.trainer import train as run_train
    train_result = run_train(repo_slug, epochs=epochs, lr=lr, batch_size=batch_size)

    console.print(f"\n  Trained {train_result['examples']} examples in {train_result['elapsed_s']}s")
    console.print(f"  Final loss: {train_result['final_loss']:.4f}")
    console.print(f"  Adapter saved: {train_result['adapter_path']}")

    console.print("\n[bold]Eval: base vs fine-tuned...[/bold]")
    from src.train.eval_harness import run_eval
    eval_report = run_eval(repo_slug)

    if "error" not in eval_report:
        console.print(f"  Base avg score: {eval_report['base_avg_score']}")
        console.print(f"  FT avg score:   {eval_report['ft_avg_score']}")
        console.print(f"  Delta: {eval_report['delta']:+.3f}")
        if eval_report["ft_beats_base"]:
            console.print("  [green]Fine-tuned beats base![/green]")
        else:
            console.print("  [yellow]FT did not beat base on this metric — RAG still carries accuracy.[/yellow]")

    console.print("\n[bold]Populating QA-augmented index...[/bold]")
    qa_path = data_dir / "qa_questions.jsonl"
    if qa_path.exists():
        from src.rag.store import populate_qa_index
        qa_count = populate_qa_index(repo_slug, qa_path)
        console.print(f"  Indexed {qa_count} questions into QA collection")
    else:
        console.print("  [yellow]No qa_questions.jsonl found — skipping QA index.[/yellow]")

    console.print("\n[bold green]Training complete![/bold green]")
    console.print(f"  Adapter: {train_result['adapter_path']}")
    console.print(f"  Eval report: {data_dir / 'eval_report.json'}")
    console.print(f"  Run `repomate serve {repo_slug} --hybrid` to use the fine-tuned model.")


@app.command()
def serve(
    repo_slug: str = typer.Argument(None),
    hybrid: bool = typer.Option(False, "--hybrid", help="Use fine-tuned adapter if available"),
) -> None:
    """Launch vLLM + FastAPI + UI. Default: RAG-only (base model)."""
    import os
    import signal
    import subprocess
    import sys
    import time

    settings = get_settings()

    if repo_slug:
        chunks_path = settings.data_dir / repo_slug / "chunks.jsonl"
        if not chunks_path.exists():
            console.print(f"[red]No indexed repo '{repo_slug}' — run `repomate init` first.[/red]")
            raise typer.Exit(code=1)
        os.environ["REPOMATE_DEFAULT_REPO"] = repo_slug

    adapter_dir = None
    if hybrid:
        if repo_slug:
            adapter_path = settings.data_dir / repo_slug / "adapters"
            if adapter_path.exists() and any(adapter_path.iterdir()):
                adapter_dir = str(adapter_path)
                console.print("[green]Hybrid mode: adapter found.[/green]")
            else:
                console.print(
                    "[yellow]No fine-tuned adapter found — running in RAG-only mode. "
                    "Run `repomate train` to enable hybrid mode.[/yellow]"
                )
                hybrid = False

    procs: list[subprocess.Popen] = []

    def shutdown(*args):
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    console.print("\n[bold]Starting vLLM server...[/bold]")
    from src.serve.vllm_server import start_vllm_server, wait_for_ready, is_ready
    from src.config import get_device

    has_gpu = get_device() != "cpu"
    if has_gpu and not is_ready():
        start_vllm_server(hybrid=hybrid, adapter_dir=adapter_dir)
        console.print("  Waiting for vLLM to load model weights...")
        if not wait_for_ready(timeout=300):
            console.print("[yellow]vLLM not ready — API will fall back to cloud LLM.[/yellow]")
    elif has_gpu:
        console.print("  vLLM already running.")
    else:
        console.print("  [yellow]No GPU detected — skipping vLLM. Chat will use cloud LLM fallback.[/yellow]")
        from src.config import get_llm_api_key
        if not get_llm_api_key():
            console.print("  [red]WARNING: No REPOMATE_LLM_API_KEY set in .env — chat will not work.[/red]")
            console.print("  Add your key: echo \"REPOMATE_LLM_API_KEY=your_key\" >> .env")

    console.print("\n[bold]Starting FastAPI server...[/bold]")

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if frontend is built
    frontend_built = os.path.exists(os.path.join(_project_root, "frontend", "out", "index.html"))
    if not frontend_built:
        console.print("  [yellow]Frontend not built. Building...[/yellow]")
        build_proc = subprocess.run(
            ["npm", "run", "build"],
            cwd=os.path.join(_project_root, "frontend"),
            capture_output=True, text=True,
        )
        if build_proc.returncode != 0:
            console.print("  [red]Frontend build failed. UI will not be available.[/red]")
            console.print(f"  {build_proc.stderr[:200]}")

    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.serve.app:app",
         "--host", "0.0.0.0", "--port", str(settings.api_port)],
        cwd=_project_root,
    )
    procs.append(api_proc)
    time.sleep(2)

    console.print(f"\n[bold green]repomate is serving![/bold green]")
    console.print(f"  UI:  http://localhost:{settings.api_port}")
    console.print(f"  API: http://localhost:{settings.api_port}/health")
    console.print(f"  vLLM: http://localhost:{settings.vllm_port}")
    console.print("\n  Press Ctrl+C to stop all services.")

    for p in procs:
        p.wait()


def main() -> None:
    setup_logging()
    app()


if __name__ == "__main__":
    main()
