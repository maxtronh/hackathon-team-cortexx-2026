import { useState } from 'react';

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="navbar">
      <div className="navbar-logo">🚀 Cortexx</div>

      <ul className={`navbar-links ${open ? 'open' : ''}`}>
        <li><a href="#features">Features</a></li>
        <li><a href="#about">About</a></li>
        <li><a href="#contact">Contact</a></li>
        <li><a href="#signup" className="navbar-cta">Get Started</a></li>
      </ul>

      <button className="navbar-toggle" onClick={() => setOpen(!open)}>
        {open ? '✕' : '☰'}
      </button>
    </nav>
  );
}
