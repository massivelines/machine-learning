const plotSeries = (time, series, surface) => {
  const series1 = time.map((item, index) => ({ x: item, y: series[index] }));

  const data = { values: [series1] };

  // Render to visor
  tfvis.render.linechart(
    {
      ...surface,
      styles: {
        width: 800,
      },
    },
    data,
    {
      zoomToFit: true,
      xLabel: 'time',
      yLabel: 'value',
      fontSize: '14',
      width: 800,
      height: 480,
    }
  );
};

const plotSeries2 = (time, series1, series2, surface) => {
  const tempSeries1 = time.map((item, index) => ({
    x: item,
    y: series1[index],
  }));
  const tempSeries2 = time.map((item, index) => ({
    x: item,
    y: series2[index],
  }));
  const series = ['acutal', 'forcast'];
  const data = { values: [tempSeries1, tempSeries2], series };

  // Render to visor
  tfvis.render.linechart(
    {
      ...surface,
      styles: {
        width: 800,
      },
    },
    data,
    {
      zoomToFit: true,
      xLabel: 'time',
      yLabel: 'value',
      fontSize: '14',
      width: 800,
      height: 480,
    }
  );
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
};

const generateDataset = (time) => {
  const baseline = 10;
  const amplitude = 40;
  const slope = 0.05;
  const noise_level = 4;

  const trendLine = trend(time, slope);
  const seasonal = seasonality(time, 365, amplitude);
  const noise = white_noise(time, noise_level);

  const series = math
    .chain(trendLine)
    .add(seasonal)
    .add(baseline)
    .add(noise)
    .done();

  plotSeries(time, series, { name: 'Data Set' });

  return series;
};

const windowed_dataset = (series, window_size, batch_size, shuffle_buffer) => {
  const windows = [];

  let index = 0;
  while (index < series.length) {
    const endPos = window_size + index;

    windows.push(series.slice(index, endPos));
    index += window_size;
  }

  tf.util.shuffle(windows);

  const xData = tf.data.array(
    windows.map((item) => item.slice(0, item.length - 1))
  );
  const yData = tf.data.array(windows.map((item) => [item[item.length - 1]]));

  return tf.data.zip({ xs: xData, ys: yData }).batch(batch_size);

  // dataset = tf.data.Dataset.from_tensor_slices(series)
  // dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  // dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  // dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  // dataset = dataset.batch(batch_size).prefetch(1)
  // return dataset
};

async function main() {
  tfvis.visor().toggleFullScreen();
  // Array of months starting at 0 for 4 years (4 * 365), add 1 (not sure why) = 1461
  const time = [...Array(4 * 365 + 1).keys()];
  const series = generateDataset(time);

  const split_time = 1000;
  const time_train = time.slice(0, split_time);
  const x_train = series.slice(0, split_time);
  const time_valid = time.slice(split_time, time.length);
  const x_valid = series.slice(split_time, time.length);

  const window_size = 20;
  const batch_size = 32;
  const shuffle_buffer_size = 1000;

  const dataset = windowed_dataset(
    x_train,
    window_size,
    batch_size,
    shuffle_buffer_size
  );

  dataset.forEachAsync((e) => console.log(e.xs.data()));
  // console.log(dataset);

  // Create a sequential model
  const model = tf.sequential();

  // First layer must have an input shape defined.
  const l0 = tf.layers.dense({ units: 1, inputShape: [window_size - 1] });
  model.add(l0);

  model.compile({
    optimizer: tf.train.momentum(1e-6, 0.9),
    loss: 'meanSquaredError',
  });

  await model.fitDataset(dataset, {
    epochs: 100,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(epoch + ':' + logs.loss);
      },
    },
  });

  console.log('Layer weights :', l0.getWeights()[0].data());

  const forecast = new Array(window_size - 1).fill(0);

  for (let i = 0; i < series.length - window_size + 1; i++) {
    const predictValues = series.slice(i, i + window_size - 1);
    const value = model
      .predictOnBatch(tf.tensor(predictValues, [1, window_size - 1]))
      .dataSync();

    forecast.push(value[0]);
  }

  const results = forecast.slice(split_time);

  plotSeries2(time_valid, x_valid, results, { name: 'Forecast' });

  const mse = tf.metrics.meanAbsoluteError(tf.tensor(x_valid), tf.tensor(results));
  mse.print();
}

document.addEventListener('DOMContentLoaded', main);
