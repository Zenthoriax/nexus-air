import { useEffect, useState } from "react"

function App() {
  const [message, setMessage] = useState("Loading...")

  useEffect(() => {
    fetch("http://127.0.0.1:8000/")
      .then(res => res.json())
      .then(data => setMessage(data.message))
  }, [])

  return (
    <div style={{ padding: "40px" }}>
      <h1>NexusAir</h1>
      <p>{message}</p>
    </div>
  )
}

export default App
