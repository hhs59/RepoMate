import './globals.css';

export const metadata = {
  title: 'repomate — Code If You Can',
  description: 'Self-training repo-native coding copilot',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
