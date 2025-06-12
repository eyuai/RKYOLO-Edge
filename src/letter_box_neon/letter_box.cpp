#include <arm_neon.h>
#include <cmath>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 关键头文件

namespace py = pybind11;

using uchar = unsigned char;

struct RGB {
    uchar r, g, b;
};

void neon_fill_border(uchar* data, int width, int height, int stride,
    int left, int right, int top, int bottom, const RGB& color)
{
    uint8x16x3_t neon_color;
    neon_color.val[0] = vdupq_n_u8(color.b);
    neon_color.val[1] = vdupq_n_u8(color.g);
    neon_color.val[2] = vdupq_n_u8(color.r);

    // Fill top and bottom borders
    for (int y = 0; y < top; ++y) {
        uchar* row = data + y * stride;
        for (int x = 0; x < width; x += 16) {
            vst3q_u8(row + x * 3, neon_color);
        }
    }

    for (int y = height - bottom; y < height; ++y) {
        uchar* row = data + y * stride;
        for (int x = 0; x < width; x += 16) {
            vst3q_u8(row + x * 3, neon_color);
        }
    }

    // Fill left and right borders
    for (int y = top; y < height - bottom; ++y) {
        uchar* row = data + y * stride;
        // Left border
        for (int x = 0; x < left; x += 16) {
            vst3q_u8(row + x * 3, neon_color);
        }
        // Right border
        for (int x = width - right; x < width; x += 16) {
            vst3q_u8(row + x * 3, neon_color);
        }
    }
}
std::tuple<py::array_t<uchar>, std::map<std::string, py::object>> letter_box(
    py::array_t<uchar> im_array,
    py::object new_shape_obj,
    std::tuple<int, int, int> pad_color = { 0, 0, 0 })
{
    auto buf = im_array.request();
    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::invalid_argument("Input must be HxWx3 array");
    }

    int origin_h = buf.shape[0];
    int origin_w = buf.shape[1];
    uchar* data = static_cast<uchar*>(buf.ptr);
    int stride = buf.strides[0];

    // 解析 new_shape
    int new_h, new_w;
    if (py::isinstance<py::int_>(new_shape_obj)) {
        int size = new_shape_obj.cast<int>();
        new_h = new_w = size;
    } else {
        auto new_shape = new_shape_obj.cast<std::tuple<int, int>>();
        new_h = std::get<0>(new_shape);
        new_w = std::get<1>(new_shape);
    }

    // 计算最小缩放比例 r
    float r = std::min(static_cast<float>(new_h) / origin_h, static_cast<float>(new_w) / origin_w);
    int new_unpad_w = std::round(origin_w * r);
    int new_unpad_h = std::round(origin_h * r);

    // 计算 padding
    int dw = new_w - new_unpad_w;
    int dh = new_h - new_unpad_h;
    int left = dw / 2;
    int right = dw - left;
    int top = dh / 2;
    int bottom = dh - top;

    // 创建输出 buffer
    py::array_t<uchar> output({ new_h, new_w, 3 });
    auto out_buf = output.request();
    uchar* out_data = static_cast<uchar*>(out_buf.ptr);
    int out_stride = out_buf.shape[1] * 3;

    // 进行双线性插值缩放
    auto resize_bilinear = [&](int src_w, int src_h, int dst_w, int dst_h) {
        float x_ratio = static_cast<float>(src_w - 1) / (dst_w > 1 ? dst_w - 1 : 1);
        float y_ratio = static_cast<float>(src_h - 1) / (dst_h > 1 ? dst_h - 1 : 1);

        for (int y = 0; y < dst_h; ++y) {
            uchar* out_row = out_data + (y + top) * out_stride + left * 3;
            for (int x = 0; x < dst_w; ++x) {
                float src_x = x * x_ratio;
                float src_y = y * y_ratio;

                int x1 = static_cast<int>(src_x);
                int y1 = static_cast<int>(src_y);
                int x2 = std::min(x1 + 1, src_w - 1);
                int y2 = std::min(y1 + 1, src_h - 1);

                float x_diff = src_x - x1;
                float y_diff = src_y - y1;

                for (int c = 0; c < 3; ++c) {
                    uchar a = data[(y1 * src_w + x1) * 3 + c];
                    uchar b = data[(y1 * src_w + x2) * 3 + c];
                    uchar c1 = data[(y2 * src_w + x1) * 3 + c];
                    uchar d = data[(y2 * src_w + x2) * 3 + c];

                    float value = a * (1 - x_diff) * (1 - y_diff) + b * x_diff * (1 - y_diff) + c1 * (1 - x_diff) * y_diff + d * x_diff * y_diff;
                    out_row[x * 3 + c] = static_cast<uchar>(value);
                }
            }
        }
    };

    if (new_unpad_w > 0 && new_unpad_h > 0) {
        resize_bilinear(origin_w, origin_h, new_unpad_w, new_unpad_h);
    }

    // 用 NEON 进行填充
    RGB color;
    color.r = std::get<0>(pad_color);
    color.g = std::get<1>(pad_color);
    color.b = std::get<2>(pad_color);
    neon_fill_border(out_data, new_w, new_h, out_stride, left, right, top, bottom, color);

    // 计算宽高缩放比例（与 Python 代码一致）
    // float w_ratio = static_cast<float>(new_w) / origin_w;
    // float h_ratio = static_cast<float>(new_h) / origin_h;

    // 构造返回的 info 结构
    std::map<std::string, py::object> info = {
        { "origin_shape", py::make_tuple(origin_h, origin_w) },
        { "new_shape", py::make_tuple(new_h, new_w) },
        { "w_ratio", py::cast(r) },
        { "h_ratio", py::cast(r) },
        { "r", py::cast(r) },
        { "dw", py::cast(dw) },
        { "dh", py::cast(dh) },
        { "pad_color", py::cast(pad_color) }
    };

    return std::make_tuple(output, info);
}

PYBIND11_MODULE(letter_box_neon, m)
{
    m.def("letter_box", &letter_box,
        "Letterbox implementation with NEON optimization",
        py::arg("im"),
        py::arg("new_shape"),
        py::arg("pad_color") = std::make_tuple(0, 0, 0),
        py::return_value_policy::automatic); // 显式指定返回策略
}