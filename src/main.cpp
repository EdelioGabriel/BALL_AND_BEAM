#include <Arduino.h>
#include <Wire.h>
#include <VL53L0X.h>
#include <ESP32Servo.h>

// ====================================================================
// PINOS
// ====================================================================
const int PIN_SDA   = 8;
const int PIN_SCL   = 9;
const int PIN_SERVO = 18;

// ====================================================================
// OBJETOS
// ====================================================================
VL53L0X sensor;
Servo   servoMotor;

// ====================================================================
// VARIÁVEIS GLOBAIS
// ====================================================================
volatile float setpoint        = 15.0f;
volatile float state[2]        = {0.0f, 0.0f};
volatile float dist_anterior   = 0.0f;
volatile float integrator      = 0.0f;
bool           controladorAtivo = false;

const float          Ts            = 0.05f;
const unsigned long  INTERVALO_US  = 50000UL;
unsigned long        tempoAnteriorMicros = 0;

const float K[2] = { -2.58360110849265f, -0.80914289088315f };
const float Ki   = -0.39478031237442f;

char    serialBuffer[32];
uint8_t bufferIndex = 0;

// ====================================================================
// COLETA DE DADOS
// ====================================================================
bool modoColeta = false;

// ====================================================================
// PROTÓTIPOS
// ====================================================================
float calcularControle();
void  processarComando();

// ====================================================================
// SETUP
// ====================================================================
void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("Iniciando...");

  Wire.begin(PIN_SDA, PIN_SCL);

  if (!sensor.init()) {
    Serial.println("Erro: sensor VL53L0X nao encontrado!");
    while (1) { delay(100); }
  }
  sensor.setMeasurementTimingBudget(50000);
  sensor.startContinuous();

  ESP32PWM::allocateTimer(0);
  servoMotor.setPeriodHertz(50);
  servoMotor.attach(PIN_SERVO, 500, 2400);
  servoMotor.write(120);

  Serial.println("Sistema Pronto. [A] Ativar | [D] Desativar | [R] Reset");
}

// ====================================================================
// LOOP PRINCIPAL
// ====================================================================
void loop() {
  // 1. Processamento serial
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (bufferIndex > 0) processarComando();
    } else if (bufferIndex < sizeof(serialBuffer) - 1) {
      serialBuffer[bufferIndex++] = c;
    }
  }

  // 2. Malha de controle a cada 50 ms
  unsigned long tempoAtual = micros();
  if (tempoAtual - tempoAnteriorMicros >= INTERVALO_US) {
    tempoAnteriorMicros = tempoAtual;

    // --- Leitura do sensor ---
    uint16_t leitura_raw_mm = sensor.readRangeContinuousMillimeters();
    if (!sensor.timeoutOccurred() && leitura_raw_mm < 2000) {
      float dist_cm = leitura_raw_mm / 10.0f;
      state[0] = (0.6237f * dist_cm) - 2.6195f;
      if (state[0] < 0.0f) state[0] = 0.0f;
    }

    // --- Velocidade (derivada discreta) ---
    state[1]      = (state[0] - dist_anterior) / Ts;
    dist_anterior = state[0];

    // --- Sinal de controle ---
    float u          = calcularControle();
    int   angulo_final = 120;

    // --- Atuação ---
    if (controladorAtivo) {
      angulo_final = (int)(120.0f - u);
      if (angulo_final > 180) angulo_final = 180;
      if (angulo_final <   0) angulo_final =   0;
      servoMotor.write(angulo_final);
    } else {
      servoMotor.write(120);
      integrator = 0.0f;
    }

    // --- Monitoramento serial ---
    if (modoColeta && controladorAtivo) {
      // CSV: tempo_s, posicao_cm, velocidade_cms, u_K
      float u_K = -K[0] * state[0] - K[1] * state[1];
      Serial.printf("%.3f,%.4f,%.4f,%.4f\n",
        tempoAnteriorMicros / 1e6f,
        state[0],
        state[1],
        u_K);
    } else {
      // print de debug original
      Serial.print("Ref:");   Serial.print(setpoint);
      Serial.print(" | Pos:"); Serial.print(state[0]);
      Serial.print(" | U:");   Serial.print(-K[0]*state[0] - K[1]*state[1] + Ki*integrator);
      Serial.print(" | Ang:"); Serial.println(120 - (int)(-K[0]*state[0] - K[1]*state[1] + Ki*integrator));
    }
  }
}

// ====================================================================
// FUNÇÕES DE SUPORTE
// ====================================================================

float calcularControle() {
  float erro = setpoint - state[0];

  if (controladorAtivo) {
    integrator += erro * Ts;
  }

  float u_K = -K[0] * state[0] - K[1]* state[1];
  float u_int = Ki * integrator;
  float u_out = u_K + u_int;

  if (u_out >  60.0f) u_out =  60.0f;
  if (u_out < -60.0f) u_out = -60.0f;

  return u_out;
}

void processarComando() {
  serialBuffer[bufferIndex] = '\0';
  char cmd = serialBuffer[0];

  if (isAlpha((unsigned char)cmd)) {
    cmd = (char)toupper((unsigned char)cmd);

    if (cmd == 'A') {
      controladorAtivo = true;
      Serial.println("\n>>> CONTROLADOR ATIVADO");
    } else if (cmd == 'D') {
      controladorAtivo = false;
      Serial.println("\n>>> CONTROLADOR DESATIVADO");
    } else if (cmd == 'R') {
      controladorAtivo = false;
      integrator   = 0.0f;
      setpoint     = 15.0f;
      state[1]     = 0.0f;
      Serial.println("\n>>> SISTEMA RESETADO");
    } else if (cmd == 'C') {
      modoColeta = !modoColeta;
      Serial.println(modoColeta ? "\n>>> MODO COLETA ON" : "\n>>> MODO COLETA OFF");
    }
  } else {
    float val = atof(serialBuffer);
    if (val > 0.0f) {
      setpoint = val;
      Serial.print("\n>>> NOVA REF: ");
      Serial.println(setpoint);
    }
  }

  bufferIndex = 0;
}