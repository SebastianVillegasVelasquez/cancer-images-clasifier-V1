"use client";
import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Loader2, SendHorizontal } from "lucide-react";

export default function UploadComponent() {
const [selectedImage, setSelectedImage] = useState<File | null>(null);
const [preview, setPreview] = useState<string | null>(null);
const [loading, setLoading] = useState(false);
const [progress, setProgress] = useState(0);
const [response, setResponse] = useState("");

  // Preview de imagen
  useEffect(() => {
    if (!selectedImage) {
      setPreview(null);
      return;
    }
    
    const objectUrl = URL.createObjectURL(selectedImage);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedImage]);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
  const file = e.target.files?.[0];
  if (file) setSelectedImage(file);
};

  const handleSubmit = async () => {
    if (!selectedImage) {
      alert("Please upload an image first.");
      return;
    }

    setLoading(true);
    setResponse("");
    setProgress(0);

    // Simulación de carga con porcentaje
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setLoading(false);

          // Simulación de respuesta del modelo
          setResponse(
            "✅ The model predicts a 78% probability of Basal Cell Carcinoma."
          );
          return 100;
        }
        return prev + 2;
      });
    }, 80);

    // 🔒 Cuando tengas el backend listo, descomenta esto:
    /*
    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      const res = await fetch("https://your-backend-url/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResponse(data.message); // Ajusta según el JSON devuelto
    } catch (error) {
      console.error("Prediction error:", error);
      setResponse("⚠️ An error occurred while analyzing the image.");
    } finally {
      setLoading(false);
    }
    */
  };

  return (
    <div className="flex flex-col items-center space-y-6">
      {/* Input file */}
      <label
        htmlFor="imageUpload"
        className="cursor-pointer flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-muted-foreground/40 rounded-xl hover:border-primary transition-colors"
      >
        {preview ? (
          <img
            src={preview}
            alt="Preview"
            className="h-full object-contain rounded-lg"
          />
        ) : (
          <div className="text-center text-muted-foreground">
            <p className="text-lg">Click or drag an image to upload</p>
            <p className="text-sm">Supported formats: JPG, PNG</p>
          </div>
        )}
        <input
          id="imageUpload"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleUpload}
        />
      </label>

      {/* Submit button */}
      <button
        onClick={handleSubmit}
        disabled={loading}
        className="bg-primary text-white px-6 py-3 rounded-xl flex items-center gap-2 font-medium hover:bg-primary/90 transition disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? (
          <>
            <Loader2 className="animate-spin w-5 h-5" />
            Analyzing...
          </>
        ) : (
          <>
            <SendHorizontal className="w-5 h-5" />
            Analyze Image
          </>
        )}
      </button>

      {/* Loading animation */}
      {loading && (
        <div className="flex flex-col items-center space-y-3 mt-6">
          <motion.div
            className="w-20 h-20 rounded-full border-4 border-primary border-t-transparent animate-spin"
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 1 }}
          />
          <p className="text-sm text-muted-foreground font-medium">
            Processing... {progress}%
          </p>
        </div>
      )}

      {/* Response bubble */}
      {response && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-xl w-full mt-8 bg-muted p-4 rounded-2xl text-left shadow-md border border-border"
        >
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
              AI
            </div>
            <p className="text-sm md:text-base text-muted-foreground leading-relaxed">
              {response}
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
}
