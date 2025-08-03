#include <Servo.h>

Servo gripperServo;
Servo rotateServo;

int gripperPin = 9;  // 夹爪舵机
int rotatePin = 10;  // 旋转舵机

int centerAngle = 90;  // 初始角度
int currentAngle = centerAngle;

void setup() {
  Serial.begin(9600);
  gripperServo.attach(gripperPin);
  rotateServo.attach(rotatePin);

  // 初始状态
  gripperServo.write(90);          // 半开
  rotateServo.write(centerAngle);  // 正中
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("ROTATE:")) {
      int delta = cmd.substring(7).toInt();
      currentAngle += delta;
      currentAngle = constrain(currentAngle, 40, 140);  // 限定角度范围
      rotateServo.write(currentAngle);
    }
    else if (cmd == "GRIP:OPEN") {
      gripperServo.write(30);  // 打开夹爪
    }
    else if (cmd == "GRIP:CLOSE") {
      gripperServo.write(90);  // 关闭夹爪
    }
  }
}
