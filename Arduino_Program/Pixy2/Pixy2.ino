#include <Pixy2.h>
#include <PIDLoop.h>

Pixy2 pixy;
PIDLoop panLoop(0.05, 0, 0.01, true); // 这里的PID参数你可以调试

const int SV_PIN = 6;  // 抓取舵机控制引脚
int32_t grip = 250;    // 抓取舵机初始值
int32_t pan = 700;     // 舵机初始位置
int32_t tilt = 600;    // 固定不动，可以后续改进

void setup() {
  Serial.begin(9600);
  pixy.init();

  // 舵机初始位置
  pixy.setServos(pan, tilt);
  analogWrite(SV_PIN, grip);
}

void loop() {

  // 5. 处理串口开合指令，只控制抓取
  if (Serial.available() > 0) {
    char var = Serial.read();

    switch (var) {
      case '1': // 闭合
        if (grip < 215) grip += 20;
        analogWrite(SV_PIN, grip);
        break;
      case '2': // 打开
        if (grip > 130) grip -= 20;
        else if (grip > 110) grip -= 5;
        analogWrite(SV_PIN, grip);
        break;
    }
  }

  // 1. Pixy2 读取目标数据
  pixy.ccc.getBlocks();

  if (pixy.ccc.numBlocks) {
    int targetIndex = -1;
    for(int i = 0; i < pixy.ccc.numBlocks; i++) {
      if (pixy.ccc.blocks[i].m_signature == 1) {  // 1代表红色签名 2代表蓝色签名
        targetIndex = i;
        break;
      }
    }
    
    // 如果找到目标，进行追踪
    if (targetIndex != -1) {
      // 2. 获取目标的中心X坐标
      int targetX = pixy.ccc.blocks[targetIndex].m_x;
      int targetY = pixy.ccc.blocks[targetIndex].m_y;
      
      int centerX = 159; // Pixy2图像中心

      // 固定步长
      int step = 10;
      if (targetX < centerX - 5) { // 目标在左边
        pan += step;
        if (pan > 1000) pan = 1000;
      } else if (targetX > centerX + 5) { // 目标在右边
        pan -= step;
        if (pan < 400) pan = 400;
      } else {
      }
      pixy.setServos(pan, tilt);

    }
  } else {
    
    }
}