const _ = require('lodash');

const loadData = () => {
  const randoms = _.range(0, 999999);
  // return randoms;
  // return randoms;
};

const data = loadData();

debugger;

/**
 * node --inspect-brk memory.js
 * chrome://inspect/#devices
 * Inside chrome
 * advance to debugger line
 * Memory tab > Heap Snapshot > Take snapshot
 * Sort shallow size descending
 * find "data" variable inside "(array)", should be first or second
 * Screenshot
 * then comment out return randoms; above
 * rerun file with inspector
 * should be lower memory usage 
 * - JavaScript Garbage Collector
 * - Shallow vs Retained Memory Usage
 */