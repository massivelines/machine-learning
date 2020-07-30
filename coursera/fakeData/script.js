// Create synthetic data
const plotSeries = (time, series, surface) => {
  const series1 = time.map((item, index) => ({ x: item, y: series[index] }));

  const data = { values: [series1] };

  // Render to visor
  tfvis.render.linechart({ ...surface }, data, {
    // zoomToFit: true,
    xLabel: 'time',
    yLabel: 'value',
    fontSize: '14',
    width: 640,
    height: 480,
  });
};

const trend = (time, slope = 0) => {
  return math.multiply(time, slope);
};

const seasonal_pattern = (season_time) => {
  return math.map(season_time, (item) => {
    if (item < 0.4) {
      return Math.cos(item * 2 * Math.PI);
    } else {
      return 1 / math.exp(3 * item);
    }
  });
};

const seasonality = (time, period, amplitude = 1, phase = 0) => {
  const season_time = math.map(time, (item) => {
    return ((item + phase) % period) / period;
  });

  return math.multiply(amplitude, seasonal_pattern(season_time));
};

const white_noise = (time, noise_level = 1) => {
  let gaussian_previous = false;

  const randomGaussian = (mean, sd) => {
    let y1, x1, x2, w;
    if (gaussian_previous) {
      y1 = y2;
      gaussian_previous = false;
    } else {
      do {
        x1 = Math.random() * 2 - 1;
        x2 = Math.random() * 2 - 1;
        w = x1 * x1 + x2 * x2;
      } while (w >= 1);
      w = Math.sqrt((-2 * Math.log(w)) / w);
      y1 = x1 * w;
      y2 = x2 * w;
      gaussian_previous = true;
    }

    const m = mean || 0;
    const s = sd || 1;
    return y1 * s + m;
  };

  const rnd = Array.apply(null, { length: time.length }).map(() =>
    randomGaussian()
  );

  return math.multiply(rnd, noise_level);

  // const rnd = math.random([1, time.length], -4, 4);
  // const flat = math.flatten(rnd);
  // return math.multiply(rnd, noise_level);
};

async function main() {
  // Array of months starting at 0 for 4 years (4 * 365), add 1 (not sure why) = 1461
  const time = [...Array(4 * 365 + 1).keys()];

  // Basic trend plot
  const trendSeries = trend(time, 0.1);
  plotSeries(time, trendSeries, { name: 'Trends Upward' });

  // Seasonality plot
  const amplitude = 40;
  const seasonalSeries = seasonality(time, 365, amplitude);
  plotSeries(time, seasonalSeries, { name: 'Seasonal Series' });

  // Trend and Seasonality
  const baseline = 10;
  const slope = 0.05;

  const trendSeasonalTrend = trend(time, slope);
  const trendSeasonalSeasonal = seasonality(time, 365, amplitude);

  const trendSeasonalSeries = math
    .chain(trendSeasonalTrend)
    .add(trendSeasonalSeasonal)
    .add(baseline)
    .done();

  plotSeries(time, trendSeasonalSeries, { name: 'Trend Seasonal Series' });

  // Noise
  const noise_level = 5;
  const noise = white_noise(time, noise_level);
  plotSeries(time, noise, { name: 'Noise' });

  // Trend Seasonality and Noise
  const dataSet = math.add(trendSeasonalSeries, noise);
  plotSeries(time, dataSet, { name: 'Data Set' });
}

// Show input data when dom is loaded
document.addEventListener('DOMContentLoaded', main);
