#include "mnist-reader.h"
#include <cassert>
#include <cstdint>
#include <iostream>

MnistReader::MnistReader(const char *label_file, const char *image_file) {
  _image_handler.open(image_file);
  char buffer[4];
  if (_image_handler.is_open()) {
    _image_handler.read(buffer, 4);
    _magic_number =
        buffer[3] | buffer[2] << 8 | buffer[1] << 16 | buffer[0] << 24;

    _image_handler.read(buffer, 4);
    _item_size = buffer[3] | buffer[2] << 8 | buffer[1] << 16 | buffer[0] << 24;

    _image_handler.read(buffer, 4);
    _rows = buffer[3] | buffer[2] << 8 | buffer[1] << 16 | buffer[0] << 24;

    _image_handler.read(buffer, 4);
    _columns = buffer[3] | buffer[2] << 8 | buffer[1] << 16 | buffer[0] << 24;

    _buffer = new char[_rows * _columns]{0x00};
    _buffer_float = new float[_rows * _columns]{0.0f};
    _buffer_int = new int[_rows * _columns]{0};

  } else {
    throw "Image file not open";
  }

  _label_handler.open(label_file);
  if (_label_handler.is_open()) {
    _label_handler.seekg(4);
    _label_handler.read(buffer, 4);
    int32_t item_size =
        buffer[3] | buffer[2] << 8 | buffer[1] << 16 | buffer[0] << 24;
    assert(item_size == _item_size);
  }
}

MnistReader::~MnistReader() {
  if (_image_handler.is_open()) {
    _image_handler.close();
  }
  if (_label_handler.is_open()) {
    _label_handler.close();
  }

  delete[] _buffer;
  _buffer = nullptr;
  delete [] _buffer_float;
  _buffer_float = nullptr;
  delete[] _buffer_int;
  _buffer_int = nullptr;
}

MnistReader::DataInfo MnistReader::operator[](int index) {

  if (!_image_handler.is_open() || !_label_handler.is_open()) {
    throw("resource file is not open");
  }

  if (index >= _item_size) {
    throw "{nullptr, -1}";
  }

  int index_image = 16 + index * (_rows * _columns);
  int index_label = 8 + index;

  _image_handler.seekg(index_image, _image_handler.beg);
  _label_handler.seekg(index_label, _label_handler.beg);

  _image_handler.read(_buffer, _rows * _columns);
  _label_handler.read(&_label, 1);

  return this->preprocess();
}

MnistReader::DataInfo MnistReader::preprocess() {
    DataInfo info;
    info.label = _label;
    info.size = _columns * _rows;
    info.bytes = sizeof(float) * info.size;

    for (int i = 0; i < info.size; ++i) {
        unsigned int number = (unsigned char)(_buffer[i]);
        //number = number;
        *(_buffer_int + i) = number;
        *(_buffer_float + i) = number / 255.0f;
    }
    info.data_ptr = _buffer_float;
    info.data_int_ptr = _buffer_int;
    return info;
}

int32_t MnistReader::magic_number() { return _magic_number; }

int32_t MnistReader::item_size() { return _item_size; }

int32_t MnistReader::image_rows() { return _rows; }

int32_t MnistReader::image_columns() { return _columns; }
