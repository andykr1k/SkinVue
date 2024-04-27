import { Routes, Route } from "react-router-dom";
import { LogInPage, LandingPage, Dashboard } from "./pages";

export default function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LogInPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </>
  );
}