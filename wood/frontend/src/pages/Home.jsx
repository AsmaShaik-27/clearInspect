import { useNavigate } from "react-router-dom"
import Header from "../components/Header"
import Footer from "../components/Footer"

export default function Home() {
  const navigate = useNavigate()

  return (
    <>
      <Header />

      <section className="bg-gradient-to-r from-black to-gray-900 text-white text-center py-24">
        <h2 className="text-4xl font-bold mb-6">
          AI-Based Wood Defect Detection System
        </h2>
        <p className="max-w-2xl mx-auto text-gray-400 mb-8">
          A real-time industrial inspection system powered by deep learning and
          Grad-CAM explainable AI.
        </p>
        <button
          onClick={() => navigate("/demo")}
          className="bg-teal-400 text-black px-8 py-3 rounded-lg text-lg"
        >
          Start Demo
        </button>
      </section>

      <section className="bg-gray-100 py-16 text-center">
        <h3 className="text-3xl font-bold mb-6">About The System</h3>
        <p className="max-w-3xl mx-auto text-gray-700">
          This system uses a ResNet-based CNN trained on the MVTec Wood dataset.
          It classifies defects and uses Grad-CAM to visualize model attention
          regions for explainability.
        </p>
      </section>

      <Footer />
    </>
  )
}