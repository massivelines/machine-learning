// Time Serries
// const originalDataSet = [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9]

// split originalDataSet into windowDataSet of 5
// const windowDataSet = [];
// const windowDataSetLength = 5;

// let index = 0;
// while (index < originalDataSet.length) {
//   windowDataSet.push(originalDataSet.slice(index, windowDataSetLength + index));
//   index += windowDataSetLength;
// }

// const windowDataSet = [
//   [0, 1, 2, 3, 4],
//   [1, 2, 3, 4, 5],
//   [2, 3, 4, 5, 6],
//   [3, 4, 5, 6, 7],
//   [4, 5, 6, 7, 8],
//   [5, 6, 7, 8, 9],
// ];

// split for labels
// const splitData = [];

// windowDataSet.forEach((row) => {
//   splitData.push([row.slice(0, row.length - 1), [row[row.length - 1]]]);
// });

// Randomize data, not best way but works for simple example
// const splitData = [
//   [[0, 1, 2, 3], [4]],
//   [[1, 2, 3, 4], [5]],
//   [[2, 3, 4, 5], [6]],
//   [[3, 4, 5, 6], [7]],
//   [[4, 5, 6, 7], [8]],
//   [[5, 6, 7, 8], [9]],
// ];

// Randomize chunked sets to prevent Sequence bias
// const randomData = splitData.sort(() => Math.random() - 0.5);
// const randomData = [
//   [[2, 3, 4, 5], [6]],
//   [[4, 5, 6, 7], [8]],
//   [[1, 2, 3, 4], [5]],
//   [[0, 1, 2, 3], [4]],
//   [[3, 4, 5, 6], [7]],
//   [[5, 6, 7, 8], [9]],
// ];

// Batch out the data for processing
// const batches = [];
// const batchLength = 2;

// let index = 0;
// while (index < randomData.length) {
//   batches.push(randomData.slice(index, batchLength + index));
//   index += batchLength;
// }

const batches = [
  [
    // [[ features ], [ label ]]
    [[2, 3, 4, 5], [6]],
    [[4, 5, 6, 7], [8]],
  ],
  [
    [[1, 2, 3, 4], [5]],
    [[0, 1, 2, 3], [4]],
  ],
  [
    [[3, 4, 5, 6], [7]],
    [[5, 6, 7, 8], [9]],
  ],
];
