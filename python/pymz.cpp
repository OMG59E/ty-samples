//
// Created by xingwg on 11/24/23.
//
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <fstream>
#include <utility>

#include "dcl_base.h"
#include "device.h"
#include "models/yolov3.h"
#include "models/yolov5.h"
#include "models/yolov6.h"
#include "models/yolov7.h"
#include "models/yolov8.h"

namespace py {
    // YOLOv5
    class YoloV5 {
    public:
        virtual int load(const std::string &modelPath) {
            input_.create(MAX_IMAGE_H, MAX_IMAGE_W, DCL_PIXEL_FORMAT_BGR_888_PACKED);
            return m_.load(modelPath);
        }

        virtual void set_iou_threshold(float iou_threshold) { m_.set_iou_threshold(iou_threshold); }
        virtual void set_conf_threshold(float conf_threshold) { m_.set_conf_threshold(conf_threshold); }

        /**
         *
         * @param cv_image  opencv img, format: H,W,C  BGR
         * @return
         */
        virtual std::vector<ty::detection_t> &inference(const pybind11::array &cv_image) {
            detections_.clear();
            auto buf = cv_image.request();
            if (buf.shape[0] > MAX_IMAGE_H || buf.shape[1] > MAX_IMAGE_W) {
                DCL_APP_LOG(DCL_ERROR, "input height[%d] > MAX_IMAGE_H[%d] or width[%d] > MAX_IMAGE_W[%d]",
                            buf.shape[0], MAX_IMAGE_H, buf.shape[1], MAX_IMAGE_W);
            } else {
                memcpy(input_.data, buf.ptr, buf.size * buf.itemsize);
                // update hwc
                input_.original_height = buf.shape[0];
                input_.original_width = buf.shape[1];
                input_.height = buf.shape[0];
                input_.width = buf.shape[1];
                input_.channels = buf.shape[2];
                input_.pixelFormat = DCL_PIXEL_FORMAT_BGR_888_PACKED;
                if (0 != m_.inference(input_, detections_)) {
                    DCL_APP_LOG(DCL_ERROR, "Failed to inference");
                }
            }
            return detections_;
        }

        virtual int unload() {
            input_.free();
            return m_.unload();
        }

    protected:
        ty::Mat input_;
        std::vector<ty::detection_t> detections_;

    private:
        ty::YoloV5 m_;
    };

    // YOLOv6
    class YoloV6 : public YoloV5 {
    private:
        ty::YoloV6 m_;
    };

    // YOLOv7
    class YoloV7 : public YoloV5 {
    private:
        ty::YoloV7 m_;
    };

    // YOLOv8
    class YoloV8 : public YoloV5 {
    private:
        ty::YoloV8 m_;
    };

    // YOLOv3
    class YoloV3 : public YoloV5 {
    private:
        ty::YoloV3 m_;
    };
}


PYBIND11_MODULE(pymz, m) {
    m.doc() = "Python bindings for modelzoo";

    pybind11::enum_<ty::pixelFormat_t>(m, "PixelFormat")
            .value("DCL_PIXEL_FORMAT_YUV_400", dclPixelFormat::DCL_PIXEL_FORMAT_YUV_400)
            .value("DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_420", dclPixelFormat::DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_420)
            .value("DCL_PIXEL_FORMAT_YVU_SEMIPLANAR_420", dclPixelFormat::DCL_PIXEL_FORMAT_YVU_SEMIPLANAR_420)
            .value("DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_422", dclPixelFormat::DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_422)
            .value("DCL_PIXEL_FORMAT_YVU_SEMIPLANAR_422", dclPixelFormat::DCL_PIXEL_FORMAT_YVU_SEMIPLANAR_422)
            .value("DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_444", dclPixelFormat::DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_444)
            .value("DCL_PIXEL_FORMAT_YVU_SEMIPLANAR_444", dclPixelFormat::DCL_PIXEL_FORMAT_YVU_SEMIPLANAR_444)
            .value("DCL_PIXEL_FORMAT_YUYV_PACKED_422", dclPixelFormat::DCL_PIXEL_FORMAT_YUYV_PACKED_422)
            .value("DCL_PIXEL_FORMAT_UYVY_PACKED_422", dclPixelFormat::DCL_PIXEL_FORMAT_UYVY_PACKED_422)
            .value("DCL_PIXEL_FORMAT_YVYU_PACKED_422", dclPixelFormat::DCL_PIXEL_FORMAT_YVYU_PACKED_422)
            .value("DCL_PIXEL_FORMAT_VYUY_PACKED_422", dclPixelFormat::DCL_PIXEL_FORMAT_VYUY_PACKED_422)
            .value("DCL_PIXEL_FORMAT_YUV_PACKED_444", dclPixelFormat::DCL_PIXEL_FORMAT_YUV_PACKED_444)
            .value("DCL_PIXEL_FORMAT_RGB_888_PACKED", dclPixelFormat::DCL_PIXEL_FORMAT_RGB_888_PACKED)
            .value("DCL_PIXEL_FORMAT_BGR_888_PACKED", dclPixelFormat::DCL_PIXEL_FORMAT_BGR_888_PACKED)
            .value("DCL_PIXEL_FORMAT_ARGB_8888", dclPixelFormat::DCL_PIXEL_FORMAT_ARGB_8888)
            .value("DCL_PIXEL_FORMAT_ABGR_8888", dclPixelFormat::DCL_PIXEL_FORMAT_ABGR_8888)
            .value("DCL_PIXEL_FORMAT_RGBA_8888", dclPixelFormat::DCL_PIXEL_FORMAT_RGBA_8888)
            .value("DCL_PIXEL_FORMAT_BGRA_8888", dclPixelFormat::DCL_PIXEL_FORMAT_BGRA_8888)
            .value("DCL_PIXEL_FORMAT_YUV_SEMI_PLANNER_420_10BIT", dclPixelFormat::DCL_PIXEL_FORMAT_YUV_SEMI_PLANNER_420_10BIT)
            .value("DCL_PIXEL_FORMAT_YVU_SEMI_PLANNER_420_10BIT", dclPixelFormat::DCL_PIXEL_FORMAT_YVU_SEMI_PLANNER_420_10BIT)
            .value("DCL_PIXEL_FORMAT_YVU_PLANAR_420", dclPixelFormat::DCL_PIXEL_FORMAT_YVU_PLANAR_420)
            .value("DCL_PIXEL_FORMAT_YVU_PLANAR_422", dclPixelFormat::DCL_PIXEL_FORMAT_YVU_PLANAR_422)
            .value("DCL_PIXEL_FORMAT_YVU_PLANAR_444", dclPixelFormat::DCL_PIXEL_FORMAT_YVU_PLANAR_444)
            .value("DCL_PIXEL_FORMAT_RGB_444", dclPixelFormat::DCL_PIXEL_FORMAT_RGB_444)
            .value("DCL_PIXEL_FORMAT_BGR_444", dclPixelFormat::DCL_PIXEL_FORMAT_BGR_444)
            .value("DCL_PIXEL_FORMAT_ARGB_4444", dclPixelFormat::DCL_PIXEL_FORMAT_ARGB_4444)
            .value("DCL_PIXEL_FORMAT_ABGR_4444", dclPixelFormat::DCL_PIXEL_FORMAT_ABGR_4444)
            .value("DCL_PIXEL_FORMAT_RGBA_4444", dclPixelFormat::DCL_PIXEL_FORMAT_RGBA_4444)
            .value("DCL_PIXEL_FORMAT_BGRA_4444", dclPixelFormat::DCL_PIXEL_FORMAT_BGRA_4444)
            .value("DCL_PIXEL_FORMAT_RGB_555", dclPixelFormat::DCL_PIXEL_FORMAT_RGB_555)
            .value("DCL_PIXEL_FORMAT_BGR_555", dclPixelFormat::DCL_PIXEL_FORMAT_BGR_555)
            .value("DCL_PIXEL_FORMAT_RGB_565", dclPixelFormat::DCL_PIXEL_FORMAT_RGB_565)
            .value("DCL_PIXEL_FORMAT_BGR_565", dclPixelFormat::DCL_PIXEL_FORMAT_BGR_565)
            .value("DCL_PIXEL_FORMAT_ARGB_1555", dclPixelFormat::DCL_PIXEL_FORMAT_ARGB_1555)
            .value("DCL_PIXEL_FORMAT_ABGR_1555", dclPixelFormat::DCL_PIXEL_FORMAT_ABGR_1555)
            .value("DCL_PIXEL_FORMAT_RGBA_1555", dclPixelFormat::DCL_PIXEL_FORMAT_RGBA_1555)
            .value("DCL_PIXEL_FORMAT_BGRA_1555", dclPixelFormat::DCL_PIXEL_FORMAT_BGRA_1555)
            .value("DCL_PIXEL_FORMAT_ARGB_8565", dclPixelFormat::DCL_PIXEL_FORMAT_ARGB_8565)
            .value("DCL_PIXEL_FORMAT_ABGR_8565", dclPixelFormat::DCL_PIXEL_FORMAT_ABGR_8565)
            .value("DCL_PIXEL_FORMAT_RGBA_8565", dclPixelFormat::DCL_PIXEL_FORMAT_RGBA_8565)
            .value("DCL_PIXEL_FORMAT_BGRA_8565", dclPixelFormat::DCL_PIXEL_FORMAT_BGRA_8565)
            .value("DCL_PIXEL_FORMAT_RGB_BAYER_8BPP", dclPixelFormat::DCL_PIXEL_FORMAT_RGB_BAYER_8BPP)
            .value("DCL_PIXEL_FORMAT_RGB_BAYER_10BPP", dclPixelFormat::DCL_PIXEL_FORMAT_RGB_BAYER_10BPP)
            .value("DCL_PIXEL_FORMAT_RGB_BAYER_12BPP", dclPixelFormat::DCL_PIXEL_FORMAT_RGB_BAYER_12BPP)
            .value("DCL_PIXEL_FORMAT_RGB_BAYER_14BPP", dclPixelFormat::DCL_PIXEL_FORMAT_RGB_BAYER_14BPP)
            .value("DCL_PIXEL_FORMAT_RGB_BAYER_16BPP", dclPixelFormat::DCL_PIXEL_FORMAT_RGB_BAYER_16BPP)
            .value("DCL_PIXEL_FORMAT_RGB_888_PLANAR", dclPixelFormat::DCL_PIXEL_FORMAT_RGB_888_PLANAR)
            .value("DCL_PIXEL_FORMAT_BGR_888_PLANAR", dclPixelFormat::DCL_PIXEL_FORMAT_BGR_888_PLANAR)
            .value("DCL_PIXEL_FORMAT_HSV_888_PACKAGE", dclPixelFormat::DCL_PIXEL_FORMAT_HSV_888_PACKAGE)
            .value("DCL_PIXEL_FORMAT_HSV_888_PLANAR", dclPixelFormat::DCL_PIXEL_FORMAT_HSV_888_PLANAR)
            .value("DCL_PIXEL_FORMAT_LAB_888_PACKAGE", dclPixelFormat::DCL_PIXEL_FORMAT_LAB_888_PACKAGE)
            .value("DCL_PIXEL_FORMAT_LAB_888_PLANAR", dclPixelFormat::DCL_PIXEL_FORMAT_LAB_888_PLANAR)
            .value("DCL_PIXEL_FORMAT_S8C1", dclPixelFormat::DCL_PIXEL_FORMAT_S8C1)
            .value("DCL_PIXEL_FORMAT_S8C2_PACKAGE", dclPixelFormat::DCL_PIXEL_FORMAT_S8C2_PACKAGE)
            .value("DCL_PIXEL_FORMAT_S8C2_PLANAR", dclPixelFormat::DCL_PIXEL_FORMAT_S8C2_PLANAR)
            .value("DCL_PIXEL_FORMAT_S16C1", dclPixelFormat::DCL_PIXEL_FORMAT_S16C1)
            .value("DCL_PIXEL_FORMAT_U8C1", dclPixelFormat::DCL_PIXEL_FORMAT_U8C1)
            .value("DCL_PIXEL_FORMAT_U16C1", dclPixelFormat::DCL_PIXEL_FORMAT_U16C1)
            .value("DCL_PIXEL_FORMAT_S32C1", dclPixelFormat::DCL_PIXEL_FORMAT_S32C1)
            .value("DCL_PIXEL_FORMAT_U32C1", dclPixelFormat::DCL_PIXEL_FORMAT_U32C1)
            .value("DCL_PIXEL_FORMAT_U64C1", dclPixelFormat::DCL_PIXEL_FORMAT_U64C1)
            .value("DCL_PIXEL_FORMAT_S64C1", dclPixelFormat::DCL_PIXEL_FORMAT_S64C1)
            .value("DCL_PIXEL_FORMAT_F32C1", dclPixelFormat::DCL_PIXEL_FORMAT_F32C1)
            .value("DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_440", dclPixelFormat::DCL_PIXEL_FORMAT_YUV_SEMIPLANAR_440)
            .value("DCL_PIXEL_FORMAT_YVU_SEMIPLANAR_440", dclPixelFormat::DCL_PIXEL_FORMAT_YVU_SEMIPLANAR_440)
            .value("DCL_PIXEL_FORMAT_YUV_PLANAR_444", dclPixelFormat::DCL_PIXEL_FORMAT_YUV_PLANAR_444)
            .value("DCL_PIXEL_FORMAT_BUTT", dclPixelFormat::DCL_PIXEL_FORMAT_BUTT)
            .value("DCL_PIXEL_FORMAT_UNKNOWN", dclPixelFormat::DCL_PIXEL_FORMAT_UNKNOWN)
            .export_values();

    pybind11::class_<ty::Mat>(m, "Mat")
            .def(pybind11::init<>())
            .def(pybind11::init<int, int, ty::pixelFormat_t>())
            .def("create", &ty::Mat::create)
            .def("free", &ty::Mat::free)
            .def("empty", &ty::Mat::empty)
            .def("c", &ty::Mat::c)
            .def("h", &ty::Mat::h)
            .def("w", &ty::Mat::w)
            .def("size", &ty::Mat::size)
            .def_readwrite("channels", &ty::Mat::channels)
            .def_readwrite("height", &ty::Mat::height)
            .def_readwrite("width", &ty::Mat::width)
            .def_readwrite("pixelFormat", &ty::Mat::pixelFormat);

    pybind11::class_<ty::Point>(m, "Point")
            .def(pybind11::init<>())
            .def(pybind11::init<int, int>())
            .def_readwrite("x", &ty::Point::x)
            .def_readwrite("y", &ty::Point::y);

    pybind11::class_<ty::Color>(m, "Color")
            .def(pybind11::init<>())
            .def(pybind11::init<int, int, int>())
            .def_readwrite("b", &ty::Color::b)
            .def_readwrite("g", &ty::Color::g)
            .def_readwrite("r", &ty::Color::r);

    pybind11::class_<ty::Box>(m, "Box")
            .def(pybind11::init<>())
            .def(pybind11::init<int, int, int, int>())
            .def_readwrite("x1", &ty::Box::x1)
            .def_readwrite("y1", &ty::Box::y1)
            .def_readwrite("x2", &ty::Box::x2)
            .def_readwrite("y2", &ty::Box::y2)
            .def("cx", &ty::Box::cx)
            .def("cy", &ty::Box::cy)
            .def("x", &ty::Box::x)
            .def("y", &ty::Box::y)
            .def("w", &ty::Box::w)
            .def("h", &ty::Box::h);

    pybind11::class_<ty::detection_t>(m, "Detection")
            .def(pybind11::init<>())
            .def_readwrite("conf", &ty::detection_t::conf)
            .def_readwrite("cls", &ty::detection_t::cls)
            .def_readwrite("name", &ty::detection_t::name)
            .def_readwrite("box", &ty::detection_t::box)
            .def_readwrite("contours", &ty::detection_t::contours)
            .def("get_pts", &ty::detection_t::getPts)
            .def("get_kpts", &ty::detection_t::getKpts)
            .def_readwrite("prob", &ty::detection_t::prob);

    m.def("dcl_init", &ty::deviceInit);
    m.def("dcl_finalize", &ty::deviceFinalize);

    pybind11::class_<py::YoloV5>(m, "YOLOv5")
            .def(pybind11::init<>())
            .def("load", &py::YoloV5::load)
            .def("unload", &py::YoloV5::unload)
            .def("set_iou_threshold", &py::YoloV5::set_iou_threshold)
            .def("set_conf_threshold", &py::YoloV5::set_conf_threshold)
            .def("inference", &py::YoloV5::inference);

    pybind11::class_<py::YoloV6, py::YoloV5>(m, "YOLOv6")
            .def(pybind11::init<>());

    pybind11::class_<py::YoloV7, py::YoloV5>(m, "YOLOv7")
            .def(pybind11::init<>());

    pybind11::class_<py::YoloV8, py::YoloV5>(m, "YOLOv8")
            .def(pybind11::init<>());

    pybind11::class_<py::YoloV3, py::YoloV5>(m, "YOLOv3")
            .def(pybind11::init<>());
}
