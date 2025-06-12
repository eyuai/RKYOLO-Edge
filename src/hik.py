# -- coding: utf-8 --

import sys
import os
import termios
import signal
from utils import log
import cv2, ctypes
import numpy as np
from ctypes import byref, POINTER, cast, sizeof, memset, c_ubyte
from ctypes import *
import time

# from db import get_db, add_recording, update_recording_end
# from const import RECORDINGS_DIR, RECORDING_ENABLED

sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
from MvCameraControl_class import *

g_bExit = False


def press_any_key_exit():
    fd = sys.stdin.fileno()
    old_ttyinfo = termios.tcgetattr(fd)
    new_ttyinfo = old_ttyinfo[:]
    new_ttyinfo[3] &= ~termios.ICANON
    new_ttyinfo[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
    try:
        os.read(fd, 7)
    except:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)


def signal_handler(sig, frame):
    global g_bExit
    log("You pressed Ctrl+C!")
    g_bExit = True


class CameraModuleError(Exception):
    pass


class CameraModule:
    def __init__(self):
        self.cam = MvCamera()
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.nConnectionNum = 0
        self.recording = False
        self.video_writer = None
        self.recording_id = None
        self.nWidth = 0
        self.nHeight = 0

    def initialize_sdk(self):
        MvCamera.MV_CC_Initialize()
        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        log("SDKVersion[0x%x]" % SDKVersion)

    def enum_devices(self):
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, self.deviceList)
        if ret != 0:
            raise CameraModuleError(f"enum devices fail! ret[0x{ret:x}]")

        if self.deviceList.nDeviceNum == 0:
            raise CameraModuleError("find no device!")

        log("Find %d devices!" % self.deviceList.nDeviceNum)
        for i in range(0, self.deviceList.nDeviceNum):
            mvcc_dev_info = cast(
                self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)
            ).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                log("\ngige device: [%d]" % i)
                strModeName = "".join(
                    chr(per) for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName
                )
                log("device model name: %s" % strModeName)
                nip1 = (
                    mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xFF000000
                ) >> 24
                nip2 = (
                    mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00FF0000
                ) >> 16
                nip3 = (
                    mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000FF00
                ) >> 8
                nip4 = mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000FF
                log("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                log("\nu3v device: [%d]" % i)
                strModeName = "".join(
                    chr(per)
                    for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName
                    if per != 0
                )
                log("device model name: %s" % strModeName)
                strSerialNumber = "".join(
                    chr(per)
                    for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber
                    if per != 0
                )
                log("user serial number: %s" % strSerialNumber)

    def connect_device(self, nConnectionNum=0):
        self.nConnectionNum = nConnectionNum
        stDeviceList = cast(
            self.deviceList.pDeviceInfo[int(self.nConnectionNum)],
            POINTER(MV_CC_DEVICE_INFO),
        ).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            raise CameraModuleError(f"create handle fail! ret[0x{ret:x}]")

        # ch:打开设备 | en:Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise CameraModuleError(f"open device fail! ret[0x{ret:x}]")

        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    log("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                log("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

        if ret != 0:
            log("set trigger mode fail! ret[0x%x]" % ret)
            raise CameraModuleError(f"set trigger mode fail! ret[0x{ret:x}]")
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            raise CameraModuleError(f"get payload size fail! ret[0x{ret:x}]")
        self.nPayloadSize = stParam.nCurValue
        ret = self.cam.MV_CC_GetIntValue("Width", stParam)
        if ret != 0:
            log("get width fail! nRet [0x%x]" % ret)
            raise CameraModuleError(f"get width fail! nRet [0x{ret:x}]")
            # sys.exit()
        self.nWidth = stParam.nCurValue
        ret = self.cam.MV_CC_GetIntValue("Height", stParam)
        if ret != 0:
            print("get height fail! nRet [0x%x]" % ret)
            raise CameraModuleError(f"get height fail! nRet [0x{ret:x}]")
            # sys.exit()
        self.nHeight = stParam.nCurValue

    def start_grabbing(self, frame_queue=None):
        self.cam.MV_CC_SetEnumValue("TriggerMode", 0)
        # self.cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
        # self.setitem("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
        # self.setitem("AcquisitionFrameRateEnable", False)
        # self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
        # self.cam.MV_CC_SetIntValue("BalanceWhiteAuto", 0)
        # fValue = 2000.0000  # 微秒 1/500 快门
        # self.cam.MV_CC_SetFloatValue("ExposureTime", fValue)
        # stFloatParam_ExposureTime = MVCC_FLOATVALUE()
        # memset(byref(stFloatParam_ExposureTime), 0, sizeof(MVCC_FLOATVALUE))
        # self.cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_ExposureTime)
        # self.cam.MV_CC_SetFloatValue("ResultingFrameRate", 50)
        self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 30)
        self.cam.MV_CC_SetBoolValue("TriggerCacheEnable", False)
        # self.cam.MV_CC_SetIntValueEx("Width", 2448)
        # self.cam.MV_CC_SetIntValueEx("Height", 2048)
        # self.cam.MV_CC_SetEnumValue("ImageCompressionMode", 1)
        # self.cam.MV_CC_SetIntValueEx("ImageCompressionQuality", 50)

        ret = self.cam.MV_CC_StartGrabbing()
        # logger.info(f"start_grabbing ExposureTime {stFloatParam_ExposureTime}")
        # logger.info("start_grabbing ret", ret)
        if ret != 0:
            raise CameraModuleError(f"start grabbing fail! ret[0x{ret:x}]")
        # start_time = time.time()
        # self.stOutFrame = MV_FRAME_OUT()
        # # logger.info(1/(time.time()-start_time))
        # memset(byref(self.stOutFrame), 0, sizeof(self.stOutFrame))

    def stop_grabbing(self):
        global g_bExit
        g_bExit = True
        # self.hThreadHandle.join()
        ret = self.cam.MV_CC_StopGrabbing()
        # if ret != 0 or ret != -2147483648:
        #     raise CameraModuleError(f"stop grabbing fail! ret[0x{ret:x}]")
        # 删除 self.data_buf 语句
        # del self.data_buf

    def close_device(self):
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            # raise CameraModuleError(f"close device fail! ret[0x{ret:x}]")
            log("close device fail! ret[0x%x]" % ret)
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            # raise CameraModuleError(f"destroy handle fail! ret[0x{ret:x}]")
            log("destroy handle fail! ret[0x%x]" % ret)
        # if hasattr(self, "data_buf"):
        #     del self.data_buf
        MvCamera.MV_CC_Finalize()

    def get_current_frame(self):
        """
        :param cam:     相机实例
        :active_way:主动取流方式的不同方法 分别是（getImagebuffer）（getoneframetimeout）
        :return:
        """
        # stOutFrame = self.stOutFrame

        ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 1000)
        # logger.info(1 / (time.time() - start_time))
        if ret != 0:
            log("get image fail! ret[0x%x]" % ret)
            return ret, None, None

        if ret == 0:
            # and None != stOutFrame.pBufAddr:
            frame_data = {
                "width": self.stOutFrame.stFrameInfo.nWidth,
                "height": self.stOutFrame.stFrameInfo.nHeight,
                "pixel_type": self.stOutFrame.stFrameInfo.enPixelType,
                "frame_num": self.stOutFrame.stFrameInfo.nFrameNum,
            }

            # image = None
            data = None
            # logger.info("width: %d, height: %d" % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight))
            # logger.info("pixel type: %d" % stOutFrame.stFrameInfo.enPixelType)
            if (
                self.stOutFrame.stFrameInfo.enPixelType == 17301505
                or self.stOutFrame.stFrameInfo.enPixelType == 17301512
                or self.stOutFrame.stFrameInfo.enPixelType == 17301513
                or self.stOutFrame.stFrameInfo.enPixelType == 17301514
                or self.stOutFrame.stFrameInfo.enPixelType == 17301515
            ):
                pData = (
                    c_ubyte
                    * self.stOutFrame.stFrameInfo.nWidth
                    * self.stOutFrame.stFrameInfo.nHeight
                )()
                ctypes.memmove(
                    ctypes.byref(pData),
                    self.stOutFrame.pBufAddr,
                    self.stOutFrame.stFrameInfo.nWidth
                    * self.stOutFrame.stFrameInfo.nHeight,
                )
                data = np.frombuffer(
                    bytes(pData),
                    count=int(
                        self.stOutFrame.stFrameInfo.nWidth
                        * self.stOutFrame.stFrameInfo.nHeight
                    ),
                    dtype=np.uint8,
                )
            elif (
                self.stOutFrame.stFrameInfo.enPixelType == 35127316
                or self.stOutFrame.stFrameInfo.enPixelType == 35127317
            ):
                pData = (
                    c_ubyte
                    * self.stOutFrame.stFrameInfo.nWidth
                    * self.stOutFrame.stFrameInfo.nHeight
                    * 3
                )()
                ctypes.memmove(
                    ctypes.byref(pData),
                    self.stOutFrame.pBufAddr,
                    self.stOutFrame.stFrameInfo.nWidth
                    * self.stOutFrame.stFrameInfo.nHeight
                    * 3,
                )
                data = np.frombuffer(
                    bytes(pData),
                    count=int(
                        self.stOutFrame.stFrameInfo.nWidth
                        * self.stOutFrame.stFrameInfo.nHeight
                        * 3
                    ),
                    dtype=np.uint8,
                )

            elif self.stOutFrame.stFrameInfo.enPixelType == 34603039:
                pData = (
                    c_ubyte
                    * self.stOutFrame.stFrameInfo.nWidth
                    * self.stOutFrame.stFrameInfo.nHeight
                    * 2
                )()
                ctypes.memmove(
                    ctypes.byref(pData),
                    self.stOutFrame.pBufAddr,
                    self.stOutFrame.stFrameInfo.nWidth
                    * self.stOutFrame.stFrameInfo.nHeight
                    * 2,
                )
                data = np.frombuffer(
                    bytes(pData),
                    count=int(
                        self.stOutFrame.stFrameInfo.nWidth
                        * self.stOutFrame.stFrameInfo.nHeight
                        * 2
                    ),
                    dtype=np.uint8,
                )

            else:
                print("no data[0x%x]" % ret)
                return ret, None, None
            image = image_control(data=data, stFrameInfo=self.stOutFrame.stFrameInfo)
            nRet = self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
            return ret, frame_data, image
        else:
            # logger.info("no data[0x%x]" % ret)
            return ret, None, None

    def release(self):
        self.stop_grabbing()
        self.close_device()


def get_current_frame(cam, queue, stop_event):
    """
    :param cam:     相机实例
    :active_way:主动取流方式的不同方法 分别是（getImagebuffer）（getoneframetimeout）
    :return:
    """
    stOutFrame = MV_FRAME_OUT()
    # logger.info(1/(time.time()-start_time))
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    # stOutFrame = self.stOutFrame
    while not stop_event.is_set():
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        # logger.info(1 / (time.time() - start_time))
        if ret != 0:
            log("get image fail! ret[0x%x]" % ret)
            continue
            # return ret, None, None

        if ret == 0:
            # and None != stOutFrame.pBufAddr:
            frame_data = {
                "width": stOutFrame.stFrameInfo.nWidth,
                "height": stOutFrame.stFrameInfo.nHeight,
                "pixel_type": stOutFrame.stFrameInfo.enPixelType,
                "frame_num": stOutFrame.stFrameInfo.nFrameNum,
            }

            # image = None
            data = None
            # logger.info("width: %d, height: %d" % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight))
            # logger.info("pixel type: %d" % stOutFrame.stFrameInfo.enPixelType)
            if (
                stOutFrame.stFrameInfo.enPixelType == 17301505
                or stOutFrame.stFrameInfo.enPixelType == 17301512
                or stOutFrame.stFrameInfo.enPixelType == 17301513
                or stOutFrame.stFrameInfo.enPixelType == 17301514
                or stOutFrame.stFrameInfo.enPixelType == 17301515
            ):
                pData = (
                    c_ubyte
                    * stOutFrame.stFrameInfo.nWidth
                    * stOutFrame.stFrameInfo.nHeight
                )()
                ctypes.memmove(
                    ctypes.byref(pData),
                    stOutFrame.pBufAddr,
                    stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight,
                )
                data = np.frombuffer(
                    bytes(pData),
                    count=int(
                        stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight
                    ),
                    dtype=np.uint8,
                )
            elif (
                stOutFrame.stFrameInfo.enPixelType == 35127316
                or stOutFrame.stFrameInfo.enPixelType == 35127317
            ):
                pData = (
                    c_ubyte
                    * stOutFrame.stFrameInfo.nWidth
                    * stOutFrame.stFrameInfo.nHeight
                    * 3
                )()
                ctypes.memmove(
                    ctypes.byref(pData),
                    stOutFrame.pBufAddr,
                    stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3,
                )
                data = np.frombuffer(
                    bytes(pData),
                    count=int(
                        stOutFrame.stFrameInfo.nWidth
                        * stOutFrame.stFrameInfo.nHeight
                        * 3
                    ),
                    dtype=np.uint8,
                )

            elif stOutFrame.stFrameInfo.enPixelType == 34603039:
                pData = (
                    c_ubyte
                    * stOutFrame.stFrameInfo.nWidth
                    * stOutFrame.stFrameInfo.nHeight
                    * 2
                )()
                ctypes.memmove(
                    ctypes.byref(pData),
                    stOutFrame.pBufAddr,
                    stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 2,
                )
                data = np.frombuffer(
                    bytes(pData),
                    count=int(
                        stOutFrame.stFrameInfo.nWidth
                        * stOutFrame.stFrameInfo.nHeight
                        * 2
                    ),
                    dtype=np.uint8,
                )

            else:
                print("no data[0x%x]" % ret)
                # return ret, None, None
                continue
            image = image_control(data=data, stFrameInfo=stOutFrame.stFrameInfo)
            nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
            if queue.full() is False:
                try:
                    queue.put_nowait((ret, frame_data, image))
                except Exception as e:
                    log("Queue is full, skipping frame")
            # return ret, frame_data, image
        else:
            # logger.info("no data[0x%x]" % ret)
            # return ret, None, None
            time.sleep(0.01)
            continue


# 枚举设备
def image_control(data, stFrameInfo):
    image = None
    if stFrameInfo.enPixelType == 17301505:
        image = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
    elif stFrameInfo.enPixelType == 17301512:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_GR2RGB)
    elif stFrameInfo.enPixelType == 17301513:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_RG2RGB)
    elif stFrameInfo.enPixelType == 17301514:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_GB2RGB)
    elif stFrameInfo.enPixelType == 17301515:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_BG2RGB)
    elif stFrameInfo.enPixelType == 35127316:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    elif stFrameInfo.enPixelType == 35127317:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BGR2BGR)
    elif stFrameInfo.enPixelType == 34603039:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_Y422)
    del data
    return image


def image_control2(data, stFrameInfo):
    enPixelType = stFrameInfo.get("pixel_type")
    nHeight = stFrameInfo.get("height")
    nWidth = stFrameInfo.get("width")
    image = None
    if enPixelType == 17301505:
        image = data.reshape((nHeight, nWidth))
    elif enPixelType == 17301512:
        data = data.reshape(nHeight, nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_GR2RGB)
    elif enPixelType == 17301513:
        data = data.reshape(nHeight, nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_RG2RGB)
    elif enPixelType == 17301514:
        data = data.reshape(nHeight, nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_GB2RGB)
    elif enPixelType == 17301515:
        data = data.reshape(nHeight, nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_BG2RGB)
    elif enPixelType == 35127316:
        data = data.reshape(nHeight, nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    elif enPixelType == 35127317:
        data = data.reshape(nHeight, nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BGR2BGR)
    elif enPixelType == 34603039:
        data = data.reshape(nHeight, nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_Y422)
    del data
    return image


if __name__ == "__main__":
    camera_module = None
    signal.signal(signal.SIGINT, signal_handler)
    try:
        camera_module = CameraModule()
        camera_module.enum_devices()
        camera_module.connect_device()
        camera_module.start_grabbing()

        # 获取当前帧数据
        frame_data = camera_module.get_current_frame()
        if frame_data:
            log(
                f"Frame {frame_data['frame_num']} received with size {frame_data['width']}x{frame_data['height']}"
            )

        # 开始录像
        # if RECORDING_ENABLED:
        #     camera_module.start_recording()

        # 等待Ctrl+C
        while not g_bExit:
            pass

        camera_module.stop_grabbing()
    except CameraModuleError as e:
        log(e)
    except Exception as e:
        log(f"Failed to initialize CameraModule: {e}")
    finally:
        if camera_module:
            try:
                # camera_module.stop_recording()
                camera_module.close_device()
            except CameraModuleError as e:
                log(e)
            # camera_module.finalize_sdk()
        sys.exit(1)


# def initHikCamera():
#     hik_camera = CameraModule()
#     hik_camera.initialize_sdk()
#     hik_camera.enum_devices()
#     hik_camera.connect_device()
#     # hik_camera.start_grabbing()
#     return hik_camera
