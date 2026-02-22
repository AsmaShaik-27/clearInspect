import { useRef, useState, useEffect } from "react"
import axios from "axios"
import Header from "../components/Header"
import Footer from "../components/Footer"

export default function Demo() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  const [stream, setStream] = useState(null)
  const [originalImage, setOriginalImage] = useState(null)
  const [heatmap, setHeatmap] = useState(null)
  const [prediction, setPrediction] = useState("")
  const [confidence, setConfidence] = useState("")
  const [explanation, setExplanation] = useState("")
  const [cameraOn, setCameraOn] = useState(true)

  useEffect(() => {
    startCamera()
  }, [])

  const startCamera = async () => {
    const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true })
    videoRef.current.srcObject = mediaStream
    setStream(mediaStream)
    setCameraOn(true)
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setCameraOn(false)
    }
  }

  const defectExplanation = (label) => {
    const explanations = {
      scratch: "The model detected elongated surface texture disruptions indicating a scratch defect.",
      hole: "A circular region with depth and intensity variation was identified as a hole defect.",
      liquid: "Irregular reflective patches suggest liquid contamination.",
      color: "Color inconsistency compared to normal wood surface detected.",
      combined: "Multiple defect types are present within the inspection region.",
      good: "No visible defects detected. Surface appears normal."
    }
    return explanations[label] || "Unknown defect."
  }

  const capture = async () => {
    const canvas = canvasRef.current
    const video = videoRef.current
    const context = canvas.getContext("2d")

    canvas.width = 224
    canvas.height = 224

    context.drawImage(video, 0, 0, 224, 224)

    const imageData = canvas.toDataURL("image/jpeg")
    setOriginalImage(imageData)

    canvas.toBlob(async (blob) => {
      const formData = new FormData()
      formData.append("file", blob, "frame.jpg")

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      )

      setPrediction(response.data.predicted_class)
      setConfidence(response.data.confidence.toFixed(4))
      setHeatmap(`data:image/jpeg;base64,${response.data.heatmap_image}`)
      setExplanation(defectExplanation(response.data.predicted_class))
    })
  }

  return (
    <>
       <Header />
    <div className="bg-gradient-to-r from-black to-gray-900 min-h-screen text-white py-16 px-6">
      <h2 className="text-3xl font-bold text-center mb-10">
        Live AI Defect Inspection
      </h2>

      {/* Camera Section */}
      <div className="flex justify-center gap-4 mb-6">
        {cameraOn ? (
          <button
            onClick={stopCamera}
            className="bg-red-500 px-6 py-2 rounded-lg 
              cursor-pointer 
              transition-all duration-300 
              hover:bg-red-600 
              hover:scale-105 
              active:scale-95"
          >
            Stop Camera
          </button>
        ) : (
          <button
            onClick={startCamera}
            className="bg-green-500 px-6 py-2 rounded-lg 
                      cursor-pointer 
                      transition-all duration-300 
                      hover:bg-green-600 
                      hover:scale-105 
                      active:scale-95"
        >
            Start Camera
          </button>
        )}

        <button
          onClick={capture}
          className="bg-teal-400 text-black px-6 py-2 rounded-lg 
           cursor-pointer 
           transition-all duration-300 
           hover:scale-105 
           hover:shadow-lg 
           active:scale-95"
        >
          Capture & Analyze
        </button>
      </div>

      <div className="flex justify-center">
        <video
          ref={videoRef}
          autoPlay
          className="rounded-lg w-96 border border-gray-600"
        />
      </div>

      <canvas ref={canvasRef} className="hidden" />

      {/* Results Section */}
      {originalImage && heatmap && (
        <div className="mt-12 max-w-6xl mx-auto">

          {/* Side by Side Images */}
          <div className="flex justify-center gap-10 flex-wrap">

            <div>
              <h3 className="text-xl mb-3 text-center">Captured Frame</h3>
              <img
                src={originalImage}
                className="w-96 rounded-lg border-4 border-gray-500"
              />
            </div>

            <div>
              <h3 className="text-xl mb-3 text-center">Grad-CAM Visualization</h3>
              <img
                src={heatmap}
                className="w-96 rounded-lg border-4 border-red-500"
              />
            </div>

          </div>

          {/* Explanation Section */}
          <div className="mt-10 bg-gray-800 p-6 rounded-lg text-center max-w-3xl mx-auto">
            <h3 className="text-2xl font-bold mb-2">
              Prediction: {prediction}
            </h3>
            <p className="mb-2">Confidence: {confidence}</p>
            <p className="text-gray-300">{explanation}</p>
          </div>

        </div>
      )}
    </div>
    <Footer />

  </>

  )
}