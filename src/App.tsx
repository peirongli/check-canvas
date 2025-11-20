import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ForgeryDetection from "@/pages/ForgeryDetection";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ForgeryDetection />} />
      </Routes>
    </Router>
  );
}
