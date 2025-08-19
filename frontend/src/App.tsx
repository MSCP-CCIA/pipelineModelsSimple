import { useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000"; // tu backend FastAPI

// imágenes según la clase predicha
const IMAGES = {
  iris: {
    setosa: "/img/setosa.png",
    versicolor: "/img/versicolor.png",
    virginica: "/img/virginica.png",
  },
  titanic: {
    0: "/img/no_survived.png",
    1: "/img/survived.png",
  },
  penguins: {
    Male: "/img/male.png",
    Female: "/img/female.png"
  },
};

function App() {
  const [model, setModel] = useState("iris");
  const [form, setForm] = useState({});
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post(`${API_BASE}/predict/${model}`, form);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error al predecir");
    }
  };

  const renderForm = () => {
    switch (model) {
      case "iris":
        return (
          <>
            <input name="sepal_length" placeholder="Sepal length" onChange={handleChange} />
            <input name="sepal_width" placeholder="Sepal width" onChange={handleChange} />
            <input name="petal_length" placeholder="Petal length" onChange={handleChange} />
            <input name="petal_width" placeholder="Petal width" onChange={handleChange} />
          </>
        );
      case "titanic":
        return (
          <>
            <input name="age" placeholder="Edad" onChange={handleChange} />
            <input name="pclass" placeholder="Clase (1-3)" onChange={handleChange} />
            <select name="who" onChange={handleChange}>
              <option value="man">Hombre</option>
              <option value="woman">Mujer</option>
              <option value="child">Niño</option>
            </select>
          </>
        );
      case "penguins":
        return (
          <>
            <input name="species" placeholder="Species" onChange={handleChange} />
            <input name="island" placeholder="Island" onChange={handleChange} />
            <input name="bill_length_mm" placeholder="Bill length (mm)" onChange={handleChange} />
            <input name="bill_depth_mm" placeholder="Bill depth (mm)" onChange={handleChange} />
            <input name="flipper_length_mm" placeholder="Flipper length (mm)" onChange={handleChange} />
            <input name="body_mass_g" placeholder="Body mass (g)" onChange={handleChange} />
          </>
        );
      default:
        return null;
    }
  };

  const getImage = () => {
    if (!result) return null;
    return IMAGES[model][result.prediction];
  };

  return (
    <div className="p-6 max-w-lg mx-auto text-center">
      <h1 className="text-2xl font-bold mb-4">ML Models Frontend</h1>

      <select
        className="mb-4 border rounded p-2"
        value={model}
        onChange={(e) => {
          setModel(e.target.value);
          setForm({});
          setResult(null);
        }}
      >
        <option value="iris">Iris</option>
        <option value="titanic">Titanic</option>
        <option value="penguins">Penguins</option>
      </select>

      <form onSubmit={handleSubmit} className="flex flex-col gap-2 mb-4">
        {renderForm()}
        <button className="bg-blue-500 text-white rounded p-2">Predecir</button>
      </form>

      {result && (
        <div>
          <p className="font-semibold">Predicción: {result.prediction}</p>
          <img src={getImage()} alt="Resultado" className="mt-2 mx-auto w-48 h-48 object-cover rounded-lg shadow" />
        </div>
      )}
    </div>
  );
}

export default App;
