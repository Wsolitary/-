# K210 MaixPy 0.5.x
import sensor, image, lcd, time, gc, sys
import KPU as kpu
from machine import UART
from fpioa_manager import fm
from Maix import I2S, FFT
import array

# YOLO 
input_size = (224, 224)
labels = ['5', '7', '8', '4', '6', '9', '1', '2', '3']
anchors = [0.81,1.19,0.56,0.84,0.66,0.97,0.38,0.78,1.09,1.84]
MODEL_PATH = "/sd/model-246779.kmodel"
#发包对应
CMD_MAP = {
    '1': b'@one\r\n',
    '2': b'@two\r\n',
    '3': b'@three\r\n',
    '4': b'@four\r\n',
    '5': b'@five\r\n',
    '6': b'@six\r\n',
    '7': b'@seven\r\n',
    '8': b'@eight\r\n',
    '9': b'@nine\r\n',
}
#UART协议 
def init_uart2():
    fm.register(6, fm.fpioa.UART2_TX, force=True)
    fm.register(7, fm.fpioa.UART2_RX, force=True)
    return UART(UART.UART2, 9600, 8, 0, 0, timeout=1000, read_buf_len=256)

class Comm:
    def __init__(self, uart):
        self.uart = uart
    def send_cmd(self, cmd):
        self.uart.write(cmd)

#麦克风 + FFT 参数
SAMPLE_RATE = 16000
FFT_POINTS  = 512

FREQ_LOW  = 920    # 修改检测下限
FREQ_HIGH = 1080   # 修改检测上限
ENERGY_THRESHOLD = 800
SEND_INTERVAL_MS = 500

freq_resolution = SAMPLE_RATE / FFT_POINTS
bin_low  = int(FREQ_LOW  / freq_resolution)
bin_high = int(FREQ_HIGH / freq_resolution)

last_mic_send = 0

#  I2S 初始化
fm.register(18, fm.fpioa.I2S0_SCLK, force=True)
fm.register(19, fm.fpioa.I2S0_WS, force=True)
fm.register(20, fm.fpioa.I2S0_IN_D0, force=True)

i2s = I2S(I2S.DEVICE_0)
i2s.channel_config(
    I2S.CHANNEL_0,
    I2S.RECEIVER,
    resolution=I2S.RESOLUTION_16_BIT,
    cycles=I2S.SCLK_CYCLES_32
)
i2s.set_sample_rate(SAMPLE_RATE)

# 声音检测函数
def check_950_1050hz_and_send(comm):
    global last_mic_send

    audio = i2s.record(FFT_POINTS)
    res = FFT.run(audio.to_bytes(), FFT_POINTS)
    spectrum = FFT.amplitude(res)

    energy_sum = 0
    for i in range(bin_low, bin_high + 1):
        energy_sum += spectrum[i]

    now = time.ticks_ms()
    if energy_sum > ENERGY_THRESHOLD:
        if time.ticks_diff(now, last_mic_send) > SEND_INTERVAL_MS:
            comm.send_cmd(b"@m\r\n")
            last_mic_send = now
            print(">>> 950~1050Hz detected, energy =", energy_sum)

#主函数
def main():
    #摄像头
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing(input_size)

    #硬件翻转（解决画面 & 文字颠倒）
    sensor.set_vflip(1)      # 垂直翻转
    sensor.set_hmirror(0)    # 水平镜像

    sensor.run(1)

    #LCD
    lcd.init(type=1)
    lcd.rotation(0)     
    lcd.clear(lcd.WHITE)

    #UART
    uart = init_uart2()
    comm = Comm(uart)

    # YOLO
    task = kpu.load(MODEL_PATH)
    kpu.init_yolo2(task, 0.5, 0.3, 5, anchors)

    last_sent_label = None

    try:
        while True:
            img = sensor.snapshot()

            # 🔊 声音检测
            check_950_1050hz_and_send(comm)

            t0 = time.ticks_ms()
            objects = kpu.run_yolo2(task, img)
            t = time.ticks_ms() - t0

            if objects:
                obj = objects[0]
                label = labels[obj.classid()]
                pos = obj.rect()

                img.draw_rectangle(pos, color=(255,0,0))
                img.draw_string(
                    pos[0], pos[1] - 10,
                    "%s %.2f" % (label, obj.value()),
                    scale=2,
                    color=(255,0,0)
                )

                if label != last_sent_label:
                    comm.send_cmd(CMD_MAP[label])
                    last_sent_label = label
            else:
                last_sent_label = None

            img.draw_string(0, 200, "t:%dms" % t, scale=2, color=(255,0,0))
            lcd.display(img)

    finally:
        kpu.deinit(task)
        gc.collect()

# ================== 入口 ==================
if __name__ == "__main__":
    main()
#那个广告被我删掉了，看着就烦