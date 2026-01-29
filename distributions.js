/**
 * Prior Distribution Explorer
 * A tool for visualizing probability density functions of common Bayesian priors
 */

// ============================================================================
// Mathematical Helper Functions
// ============================================================================

// Lanczos approximation for the gamma function
function gamma(z) {
  if (z < 0.5) {
    return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
  }
  z -= 1;
  const g = 7;
  const c = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];
  let x = c[0];
  for (let i = 1; i < g + 2; i++) {
    x += c[i] / (z + i);
  }
  const t = z + g + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}

// Log gamma function for numerical stability
function logGamma(z) {
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  z -= 1;
  const g = 7;
  const c = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];
  let x = c[0];
  for (let i = 1; i < g + 2; i++) {
    x += c[i] / (z + i);
  }
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

// Beta function using log gamma for stability
function beta(a, b) {
  return Math.exp(logGamma(a) + logGamma(b) - logGamma(a + b));
}

// ============================================================================
// Distribution Definitions
// ============================================================================

const distributions = {
  normal: {
    name: 'Normal',
    latex: 'f(x) \\propto \\exp\\!\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)',
    params: [
      { name: 'μ', label: 'Mean (μ)', default: 0, min: -10, max: 10, step: 0.1 },
      { name: 'σ', label: 'Std Dev (σ)', default: 1, min: 0.1, max: 5, step: 0.1 }
    ],
    pdf: (x, params) => {
      const mu = params['μ'];
      const sigma = params['σ'];
      const coef = 1 / (sigma * Math.sqrt(2 * Math.PI));
      const exp = Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
      return coef * exp;
    },
    domain: (params) => {
      const mu = params['μ'];
      const sigma = params['σ'];
      return [mu - 4 * sigma, mu + 4 * sigma];
    },
    description: 'Symmetric bell curve, commonly used for location parameters.'
  },

  halfNormal: {
    name: 'Half-Normal',
    latex: 'f(x) \\propto \\exp\\!\\left(-\\frac{x^2}{2\\sigma^2}\\right), \\; x \\geq 0',
    params: [
      { name: 'σ', label: 'Scale (σ)', default: 1, min: 0.1, max: 5, step: 0.1 }
    ],
    pdf: (x, params) => {
      if (x < 0) return 0;
      const sigma = params['σ'];
      const coef = Math.sqrt(2) / (sigma * Math.sqrt(Math.PI));
      const exp = Math.exp(-0.5 * Math.pow(x / sigma, 2));
      return coef * exp;
    },
    domain: (params) => {
      const sigma = params['σ'];
      return [0, 4 * sigma];
    },
    description: 'Positive-only normal, used for scale parameters.'
  },

  logNormal: {
    name: 'Log-Normal',
    latex: 'f(x) \\propto \\frac{1}{x}\\exp\\!\\left(-\\frac{(\\ln x - \\mu)^2}{2\\sigma^2}\\right)',
    params: [
      { name: 'μ', label: 'Log Mean (μ)', default: 0, min: -3, max: 3, step: 0.1 },
      { name: 'σ', label: 'Log Std Dev (σ)', default: 1, min: 0.1, max: 2, step: 0.1 }
    ],
    pdf: (x, params) => {
      if (x <= 0) return 0;
      const mu = params['μ'];
      const sigma = params['σ'];
      const coef = 1 / (x * sigma * Math.sqrt(2 * Math.PI));
      const exp = Math.exp(-0.5 * Math.pow((Math.log(x) - mu) / sigma, 2));
      return coef * exp;
    },
    domain: (params) => {
      const mu = params['μ'];
      const sigma = params['σ'];
      // Use quantiles to determine range
      const median = Math.exp(mu);
      const upper = Math.exp(mu + 3 * sigma);
      return [0, Math.min(upper, 20)];
    },
    description: 'Positive, right-skewed distribution.'
  },

  cauchy: {
    name: 'Cauchy',
    latex: 'f(x) \\propto \\frac{1}{1 + \\left(\\frac{x - x_0}{\\gamma}\\right)^2}',
    params: [
      { name: 'x₀', label: 'Location (x₀)', default: 0, min: -10, max: 10, step: 0.1 },
      { name: 'γ', label: 'Scale (γ)', default: 1, min: 0.1, max: 5, step: 0.1 }
    ],
    pdf: (x, params) => {
      const x0 = params['x₀'];
      const gamma = params['γ'];
      return 1 / (Math.PI * gamma * (1 + Math.pow((x - x0) / gamma, 2)));
    },
    domain: (params) => {
      const x0 = params['x₀'];
      const gamma = params['γ'];
      return [x0 - 10 * gamma, x0 + 10 * gamma];
    },
    description: 'Heavy-tailed distribution, useful for robust priors.'
  },

  halfCauchy: {
    name: 'Half-Cauchy',
    latex: 'f(x) \\propto \\frac{1}{1 + (x/\\gamma)^2}, \\; x \\geq 0',
    params: [
      { name: 'γ', label: 'Scale (γ)', default: 1, min: 0.1, max: 5, step: 0.1 }
    ],
    pdf: (x, params) => {
      if (x < 0) return 0;
      const gamma = params['γ'];
      return 2 / (Math.PI * gamma * (1 + Math.pow(x / gamma, 2)));
    },
    domain: (params) => {
      const gamma = params['γ'];
      return [0, 10 * gamma];
    },
    description: 'Positive heavy-tailed, common for variance priors.'
  },

  exponential: {
    name: 'Exponential',
    latex: 'f(x) = \\lambda e^{-\\lambda x}, \\; x \\geq 0',
    params: [
      { name: 'λ', label: 'Rate (λ)', default: 1, min: 0.1, max: 5, step: 0.1 }
    ],
    pdf: (x, params) => {
      if (x < 0) return 0;
      const lambda = params['λ'];
      return lambda * Math.exp(-lambda * x);
    },
    domain: (params) => {
      const lambda = params['λ'];
      return [0, 5 / lambda];
    },
    description: 'Memoryless distribution for positive parameters.'
  },

  gamma: {
    name: 'Gamma',
    latex: 'f(x) \\propto x^{\\alpha-1} e^{-\\beta x}, \\; x > 0',
    params: [
      { name: 'α', label: 'Shape (α)', default: 2, min: 0.1, max: 10, step: 0.1 },
      { name: 'β', label: 'Rate (β)', default: 1, min: 0.1, max: 5, step: 0.1 }
    ],
    pdf: (x, params) => {
      if (x <= 0) return 0;
      const alpha = params['α'];
      const beta = params['β'];
      const coef = Math.pow(beta, alpha) / gamma(alpha);
      return coef * Math.pow(x, alpha - 1) * Math.exp(-beta * x);
    },
    domain: (params) => {
      const alpha = params['α'];
      const beta = params['β'];
      // Use mean + 3*stddev as upper bound
      const mean = alpha / beta;
      const std = Math.sqrt(alpha) / beta;
      return [0, Math.max(mean + 4 * std, 0.1)];
    },
    description: 'Flexible positive distribution with shape control.'
  },

  beta: {
    name: 'Beta',
    latex: 'f(x) \\propto x^{\\alpha-1}(1-x)^{\\beta-1}, \\; x \\in (0,1)',
    params: [
      { name: 'α', label: 'Alpha (α)', default: 2, min: 0.1, max: 10, step: 0.1 },
      { name: 'β', label: 'Beta (β)', default: 5, min: 0.1, max: 10, step: 0.1 }
    ],
    pdf: (x, params) => {
      if (x <= 0 || x >= 1) return 0;
      const alpha = params['α'];
      const b = params['β'];
      const coef = 1 / beta(alpha, b);
      return coef * Math.pow(x, alpha - 1) * Math.pow(1 - x, b - 1);
    },
    domain: () => [0, 1],
    description: 'Bounded [0,1] distribution, ideal for probabilities.'
  },

  uniform: {
    name: 'Uniform',
    latex: 'f(x) = \\frac{1}{b-a}, \\; x \\in [a, b]',
    params: [
      { name: 'a', label: 'Min (a)', default: 0, min: -10, max: 9, step: 0.1 },
      { name: 'b', label: 'Max (b)', default: 1, min: -9, max: 10, step: 0.1 }
    ],
    pdf: (x, params) => {
      const a = params['a'];
      const b = params['b'];
      if (b <= a) return 0;
      if (x < a || x > b) return 0;
      return 1 / (b - a);
    },
    domain: (params) => {
      const a = params['a'];
      const b = params['b'];
      const padding = (b - a) * 0.1;
      return [a - padding, b + padding];
    },
    description: 'Flat prior, all values equally likely in range.'
  },

  studentT: {
    name: "Student's t",
    latex: 'f(x) \\propto \\left(1 + \\frac{1}{\\nu}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2\\right)^{-\\frac{\\nu+1}{2}}',
    params: [
      { name: 'ν', label: 'Degrees of Freedom (ν)', default: 3, min: 1, max: 30, step: 1 },
      { name: 'μ', label: 'Location (μ)', default: 0, min: -10, max: 10, step: 0.1 },
      { name: 'σ', label: 'Scale (σ)', default: 1, min: 0.1, max: 5, step: 0.1 }
    ],
    pdf: (x, params) => {
      const nu = params['ν'];
      const mu = params['μ'];
      const sigma = params['σ'];
      const z = (x - mu) / sigma;
      const coef = gamma((nu + 1) / 2) / (sigma * Math.sqrt(nu * Math.PI) * gamma(nu / 2));
      return coef * Math.pow(1 + z * z / nu, -(nu + 1) / 2);
    },
    domain: (params) => {
      const mu = params['μ'];
      const sigma = params['σ'];
      return [mu - 6 * sigma, mu + 6 * sigma];
    },
    description: 'Heavy-tailed alternative to normal, robust to outliers.'
  }
};

// ============================================================================
// Color Palette for Multiple Distributions
// ============================================================================

const colorPalette = [
  '#8b4513', // Sienna (site accent)
  '#2c5d63', // Deep teal
  '#a4616b', // Dusty rose
  '#6b6b3e', // Olive
  '#4a5568', // Slate blue
  '#c75b39', // Burnt orange
  '#5b7553', // Forest green
  '#7c5295', // Muted purple
];

// ============================================================================
// Distribution Plotter Class
// ============================================================================

class DistributionPlotter {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.distributions = []; // Active distributions
    this.nextId = 0;
    this.colorIndex = 0;

    // SVG dimensions
    this.margin = { top: 20, right: 30, bottom: 40, left: 50 };
    this.width = 600;
    this.height = 300;
    this.plotWidth = this.width - this.margin.left - this.margin.right;
    this.plotHeight = this.height - this.margin.top - this.margin.bottom;

    // Fixed axis ranges
    this.xMin = -5;
    this.xMax = 10;
    this.yMax = 1; // Will be calculated based on visible distributions

    this.init();
  }

  init() {
    this.render();
    this.updatePlot();
  }

  render() {
    this.container.innerHTML = `
      <div class="dist-plotter">
        <div class="dist-plot-area">
          <svg id="dist-svg" viewBox="0 0 ${this.width} ${this.height}"></svg>
        </div>
        <div class="dist-controls">
          <div class="dist-add-section">
            <select id="dist-type-select" class="dist-select">
              ${Object.entries(distributions).map(([key, dist]) =>
                `<option value="${key}">${dist.name}</option>`
              ).join('')}
            </select>
            <button id="dist-add-btn" class="dist-add-btn">+ Add Distribution</button>
          </div>
          <div id="dist-list" class="dist-list"></div>
        </div>
      </div>
    `;

    // Setup event listeners
    document.getElementById('dist-add-btn').addEventListener('click', () => this.addDistribution());

    this.svg = document.getElementById('dist-svg');
    this.distList = document.getElementById('dist-list');
  }

  getNextColor() {
    const color = colorPalette[this.colorIndex % colorPalette.length];
    this.colorIndex++;
    return color;
  }

  addDistribution() {
    const type = document.getElementById('dist-type-select').value;
    const distDef = distributions[type];

    // Create initial params object
    const params = {};
    distDef.params.forEach(p => {
      params[p.name] = p.default;
    });

    const dist = {
      id: this.nextId++,
      type: type,
      params: params,
      color: this.getNextColor(),
      visible: true
    };

    this.distributions.push(dist);
    this.renderDistributionCard(dist);
    this.updatePlot();
  }

  renderDistributionCard(dist) {
    const distDef = distributions[dist.type];
    const card = document.createElement('div');
    card.className = 'dist-card';
    card.id = `dist-card-${dist.id}`;
    card.innerHTML = `
      <div class="dist-card-header">
        <button class="dist-visibility-btn" data-id="${dist.id}" style="background-color: ${dist.visible ? dist.color : 'transparent'}; border-color: ${dist.color}">
        </button>
        <span class="dist-card-title">${distDef.name}</span>
        <button class="dist-remove-btn" data-id="${dist.id}">×</button>
      </div>
      <div class="dist-card-formula"></div>
      <div class="dist-card-params">
        ${distDef.params.map(p => `
          <div class="dist-param">
            <label class="dist-param-label">${p.label}</label>
            <div class="dist-param-control">
              <input type="range"
                     class="dist-slider"
                     data-id="${dist.id}"
                     data-param="${p.name}"
                     min="${p.min}"
                     max="${p.max}"
                     step="${p.step}"
                     value="${dist.params[p.name]}">
              <input type="number"
                     class="dist-number"
                     data-id="${dist.id}"
                     data-param="${p.name}"
                     min="${p.min}"
                     max="${p.max}"
                     step="${p.step}"
                     value="${dist.params[p.name]}">
            </div>
          </div>
        `).join('')}
      </div>
      <div class="dist-card-description">${distDef.description}</div>
    `;

    // Add event listeners
    card.querySelector('.dist-visibility-btn').addEventListener('click', (e) => {
      this.toggleVisibility(parseInt(e.target.dataset.id));
    });

    card.querySelector('.dist-remove-btn').addEventListener('click', (e) => {
      this.removeDistribution(parseInt(e.target.dataset.id));
    });

    card.querySelectorAll('.dist-slider').forEach(slider => {
      slider.addEventListener('input', (e) => {
        this.updateParam(
          parseInt(e.target.dataset.id),
          e.target.dataset.param,
          parseFloat(e.target.value)
        );
      });
    });

    card.querySelectorAll('.dist-number').forEach(input => {
      input.addEventListener('change', (e) => {
        this.updateParam(
          parseInt(e.target.dataset.id),
          e.target.dataset.param,
          parseFloat(e.target.value)
        );
      });
    });

    this.distList.appendChild(card);

    // Render LaTeX formula with KaTeX
    const formulaEl = card.querySelector('.dist-card-formula');
    if (distDef.latex && typeof katex !== 'undefined') {
      katex.render(distDef.latex, formulaEl, {
        throwOnError: false,
        displayMode: false
      });
    }
  }

  updateParam(id, paramName, value) {
    const dist = this.distributions.find(d => d.id === id);
    if (dist) {
      dist.params[paramName] = value;

      // Sync slider and number input
      const card = document.getElementById(`dist-card-${id}`);
      card.querySelector(`.dist-slider[data-param="${paramName}"]`).value = value;
      card.querySelector(`.dist-number[data-param="${paramName}"]`).value = value;

      this.updatePlot();
    }
  }

  toggleVisibility(id) {
    const dist = this.distributions.find(d => d.id === id);
    if (dist) {
      dist.visible = !dist.visible;
      const btn = document.querySelector(`.dist-visibility-btn[data-id="${id}"]`);
      btn.style.backgroundColor = dist.visible ? dist.color : 'transparent';
      this.updatePlot();
    }
  }

  removeDistribution(id) {
    this.distributions = this.distributions.filter(d => d.id !== id);
    const card = document.getElementById(`dist-card-${id}`);
    if (card) card.remove();
    this.updatePlot();
  }

  calculateYMax() {
    // Calculate max y value across all visible distributions within the fixed x range
    let yMax = 0;
    const numPoints = 500;
    const xStep = (this.xMax - this.xMin) / numPoints;

    this.distributions.forEach(dist => {
      if (!dist.visible) return;
      const distDef = distributions[dist.type];

      for (let i = 0; i <= numPoints; i++) {
        const x = this.xMin + i * xStep;
        const y = distDef.pdf(x, dist.params);
        if (isFinite(y) && !isNaN(y)) {
          yMax = Math.max(yMax, y);
        }
      }
    });

    return yMax > 0 ? yMax * 1.1 : 1; // Add 10% padding
  }

  updatePlot() {
    const xMin = this.xMin;
    const xMax = this.xMax;
    const numPoints = 500;
    const xStep = (xMax - xMin) / numPoints;

    // Calculate yMax from visible distributions
    const yMax = this.calculateYMax();

    // Generate paths for all visible distributions
    const allPaths = [];
    this.distributions.forEach(dist => {
      if (!dist.visible) return;

      const distDef = distributions[dist.type];
      const points = [];

      for (let i = 0; i <= numPoints; i++) {
        const x = xMin + i * xStep;
        const y = distDef.pdf(x, dist.params);
        if (isFinite(y) && !isNaN(y)) {
          points.push({ x, y });
        }
      }

      allPaths.push({ points, color: dist.color });
    });

    // Scale functions
    const scaleX = (x) => this.margin.left + ((x - xMin) / (xMax - xMin)) * this.plotWidth;
    const scaleY = (y) => this.margin.top + this.plotHeight - (y / yMax) * this.plotHeight;

    // Build SVG
    let svgContent = '';

    // Define clipping path
    svgContent += `<defs>
      <clipPath id="plot-clip">
        <rect x="${this.margin.left}" y="${this.margin.top}"
              width="${this.plotWidth}" height="${this.plotHeight}"/>
      </clipPath>
    </defs>`;

    // Background
    svgContent += `<rect x="${this.margin.left}" y="${this.margin.top}"
                         width="${this.plotWidth}" height="${this.plotHeight}"
                         fill="#fff"/>`;

    // Grid lines
    const numGridLines = 5;
    for (let i = 0; i <= numGridLines; i++) {
      const y = this.margin.top + (i / numGridLines) * this.plotHeight;
      svgContent += `<line x1="${this.margin.left}" y1="${y}"
                           x2="${this.margin.left + this.plotWidth}" y2="${y}"
                           stroke="#ddd" stroke-dasharray="3,3" opacity="0.5"/>`;
    }

    // X-axis
    svgContent += `<line x1="${this.margin.left}" y1="${this.margin.top + this.plotHeight}"
                         x2="${this.margin.left + this.plotWidth}" y2="${this.margin.top + this.plotHeight}"
                         stroke="#666" stroke-width="1"/>`;

    // Y-axis
    svgContent += `<line x1="${this.margin.left}" y1="${this.margin.top}"
                         x2="${this.margin.left}" y2="${this.margin.top + this.plotHeight}"
                         stroke="#666" stroke-width="1"/>`;

    // X-axis ticks and labels
    const xTicks = this.calculateTicks(xMin, xMax, 6);
    xTicks.forEach(tick => {
      const x = scaleX(tick);
      svgContent += `<line x1="${x}" y1="${this.margin.top + this.plotHeight}"
                           x2="${x}" y2="${this.margin.top + this.plotHeight + 5}"
                           stroke="#666"/>`;
      svgContent += `<text x="${x}" y="${this.margin.top + this.plotHeight + 20}"
                           text-anchor="middle" fill="#666"
                           font-size="11" font-family="Georgia, serif">${this.formatTick(tick)}</text>`;
    });

    // Y-axis ticks and labels
    const yTicks = this.calculateTicks(0, yMax, 5);
    yTicks.forEach(tick => {
      const y = scaleY(tick);
      svgContent += `<line x1="${this.margin.left - 5}" y1="${y}"
                           x2="${this.margin.left}" y2="${y}"
                           stroke="#666"/>`;
      svgContent += `<text x="${this.margin.left - 10}" y="${y + 4}"
                           text-anchor="end" fill="#666"
                           font-size="11" font-family="Georgia, serif">${this.formatTick(tick)}</text>`;
    });

    // Axis labels
    svgContent += `<text x="${this.margin.left + this.plotWidth / 2}" y="${this.height - 5}"
                         text-anchor="middle" fill="#444"
                         font-size="12" font-family="Georgia, serif">x</text>`;
    svgContent += `<text x="15" y="${this.margin.top + this.plotHeight / 2}"
                         text-anchor="middle" fill="#444"
                         font-size="12" font-family="Georgia, serif"
                         transform="rotate(-90, 15, ${this.margin.top + this.plotHeight / 2})">Density</text>`;

    // Draw distribution paths (clipped to plot area)
    allPaths.forEach(({ points, color }) => {
      if (points.length < 2) return;

      const pathData = points.map((p, i) =>
        `${i === 0 ? 'M' : 'L'} ${scaleX(p.x).toFixed(2)} ${scaleY(p.y).toFixed(2)}`
      ).join(' ');

      svgContent += `<path d="${pathData}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" clip-path="url(#plot-clip)"/>`;
    });

    this.svg.innerHTML = svgContent;
  }

  calculateTicks(min, max, count) {
    const range = max - min;
    const roughStep = range / count;
    const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
    const residual = roughStep / magnitude;

    let step;
    if (residual > 5) step = 10 * magnitude;
    else if (residual > 2) step = 5 * magnitude;
    else if (residual > 1) step = 2 * magnitude;
    else step = magnitude;

    const ticks = [];
    const start = Math.ceil(min / step) * step;
    for (let tick = start; tick <= max; tick += step) {
      ticks.push(tick);
    }
    return ticks;
  }

  formatTick(value) {
    if (Math.abs(value) < 0.001 && value !== 0) {
      return value.toExponential(1);
    }
    if (Math.abs(value) >= 1000) {
      return value.toExponential(1);
    }
    // Remove trailing zeros
    return parseFloat(value.toPrecision(3)).toString();
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('distribution-plotter')) {
    window.plotter = new DistributionPlotter('distribution-plotter');
  }
});
