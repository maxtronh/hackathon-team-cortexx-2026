import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Features from './components/Features';
import './App.css';

function App() {
  return (
    <div className="app">
      <Navbar />
      <main>
        <Hero />
        <Features />
      </main>
    </div>
  );
}

export default App;
