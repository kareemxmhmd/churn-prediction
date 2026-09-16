import "./globals.css";

export const metadata = {
  title: "Churn Prediction",
  description: "Predict customer churn probability",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
