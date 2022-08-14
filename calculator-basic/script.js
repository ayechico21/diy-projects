const screen = document.getElementsByClassName("calc-screen")[0];
let buffer = "0"; // screen content
let previousOperator = undefined; // operator before a number
let total = 0;

/**
 * refresh content of calculator screen
 */
function rerender(buffer) {
  screen.innerText = buffer;
}

function getOperation(button) {
  if (isNaN(button)) {
    handleOperator(button); // input is a operator
  } else {
    handleNumber(button); // input is a number
  }
  rerender(buffer); // refresh screen after each input
}

function handleOperator(button) {
  if (button === "C") { // clear screen content
    buffer = "0"; 
  } else if (button === "←") { // backspace screen content
    if (buffer.length === 1) { // only one character on screen
      buffer = "0";
    } else {
      buffer = buffer.substring(0, buffer.length - 1);
    }
  } else if (button === "=") {
    if (previousOperator === undefined) return; // any operation require one operator
    doMath(); 
    previousOperator = undefined; // operation done, disard operator
    buffer = "" + total; // update screen content
    total = 0; // calculation done, reset
  } else {
    handleMath(button);
  }
}

function handleNumber(button) {
  if (buffer === "0") { // single digit number
    buffer = button;
  } else {
    buffer += button; // multi digit number
  }
}
function handleMath(button) {
  if (buffer === "0") return; // no operand
  if (total === 0) { // first operand
    total = parseInt(buffer);
  } else {
    doMath();
  }
  previousOperator = button; // required for next operation
  buffer = "0"; // clear screen content
}

function doMath() {
  intBuffer = parseInt(buffer); // operations only on integers
  if (previousOperator === "+") total += intBuffer;
  else if (previousOperator === "-") total -= intBuffer;
  else if (previousOperator === "*") total *= intBuffer;
  else if (previousOperator === "/") total /= intBuffer;
}

function main() {
  document.querySelector(".calc").addEventListener("click", function (event) {
    getOperation(event.target.innerText);
  });
}

main();
