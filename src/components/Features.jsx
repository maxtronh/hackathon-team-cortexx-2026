const cards = [
  {
    icon: '⚡',
    title: 'Lightning Fast',
    description:
      'Optimized for speed so your users never have to wait. Sub-second load times out of the box.',
  },
  {
    icon: '🔒',
    title: 'Secure by Default',
    description:
      'Enterprise-grade security baked in from day one. SOC 2 compliant and end-to-end encrypted.',
  },
  {
    icon: '📊',
    title: 'Powerful Analytics',
    description:
      'Real-time dashboards and insights that help you make data-driven decisions effortlessly.',
  },
];

export default function Features() {
  return (
    <section className="features" id="features">
      <h2>Why Choose Us</h2>
      <p className="subtitle">
        Everything you need to go from idea to production.
      </p>
      <div className="features-grid">
        {cards.map((card) => (
          <div className="feature-card" key={card.title}>
            <div className="feature-icon">{card.icon}</div>
            <h3>{card.title}</h3>
            <p>{card.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
