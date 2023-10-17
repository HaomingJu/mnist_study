#pragma once

#include <cstdint>
#include <fstream>
#include <utility>

class MnistReader {
    public:
    struct DataInfo{
        float* data_ptr = nullptr;
        int* data_int_ptr = nullptr;
        int label = -1;
        int32_t bytes = 0;
        int32_t size = 0;
    };

public:
  MnistReader(const char *label_file, const char *image_file);
  ~MnistReader();
  DataInfo operator[](int index);

  int32_t magic_number();
  int32_t item_size();
  int32_t image_rows();
  int32_t image_columns();

private:
  char* _buffer = nullptr;
  float* _buffer_float = nullptr;
  int* _buffer_int = nullptr;
  char _label = 'A';

  std::ifstream _image_handler;
  std::ifstream _label_handler;
  int32_t _magic_number = -1;
  int32_t _item_size = -1;
  int32_t _rows = -1;
  int32_t _columns = -1;

private:
  DataInfo preprocess();
};
