# Circuit Synthesis Demonstration
Show off integration with cool work coming out of BQSKit. Also, show 
python circuit synthesis with 3rd party libraries.

# Goals
![toffoli](toffoli.png)
* demo circuit synthesis language extension in C++
* Show the same in python, use different synthesis strategy. Note the use of Numpy...
* Note that decomposed circuits are cached.

Notes:
To build truth table...
```cpp
  std::vector<std::vector<int>> truth_table;
  for (int i = 0; i < 8; ++i) {
    truth_table.push_back({i / 4 % 2, i / 2 % 2, i % 2});
  }
```