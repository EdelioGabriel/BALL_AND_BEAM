#include <Arduino.h>
#include <Wire.h>
#include <VL53L0X.h>
#include <ESP32Servo.h>
#include "pinn_weights.h"

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
bool           usarPINN         = false;   // false = analítico | true = PINN

const float          Ts            = 0.05f;
const unsigned long  INTERVALO_US  = 50000UL;
unsigned long        tempoAnteriorMicros = 0;

// Ganhos do controlador analítico
const float K[2] = { -2.58360110849265f, -0.80914289088315f };
const float Ki   =  -0.39478031237442f;

char    serialBuffer[32];
uint8_t bufferIndex = 0;

// ====================================================================
// COLETA DE DADOS
// ====================================================================
bool modoColeta = false;

// ====================================================================
// INFERÊNCIA DA PINN
// ====================================================================

// Aplica tanh elemento a elemento em vetor
void tanh_vec(float* v, int n) {
  for (int i = 0; i < n; i++) {
    v[i] = tanhf(v[i]);
  }
}

// Multiplicação matriz-vetor: out = W * in + b
// W: shape (n_out, n_in), armazenada em row-major
void linear(const float* W, const float* b,
            const float* in, float* out,
            int n_in, int n_out) {
  for (int i = 0; i < n_out; i++) {
    float acc = b[i];
    for (int j = 0; j < n_in; j++) {
      acc += W[i * n_in + j] * in[j];
    }
    out[i] = acc;
  }
}

// Inferência completa da PINN
// Entrada: [posicao_cm, velocidade_cms, x_ref_cm, e_int]
// Saída:   angulo em graus
float pinnInference(float pos, float vel, float xref, float eint) {
  float h0[PINN_N_HIDDEN];
  float h1[PINN_N_HIDDEN];
  float out[PINN_N_OUTPUTS];

  float input[PINN_N_INPUTS] = {pos, vel, xref, eint};

  // Camada 0: Linear + Tanh
  linear(net_0_weight, net_0_bias, input, h0, PINN_N_INPUTS, PINN_N_HIDDEN);
  tanh_vec(h0, PINN_N_HIDDEN);

  // Camada 1: Linear + Tanh
  linear(net_2_weight, net_2_bias, h0, h1, PINN_N_HIDDEN, PINN_N_HIDDEN);
  tanh_vec(h1, PINN_N_HIDDEN);

  // Camada de saída: Linear (sem ativação)
  linear(net_4_weight, net_4_bias, h1, out, PINN_N_HIDDEN, PINN_N_OUTPUTS);

  // Limita saída ao range físico
  float theta = out[0];
  if (theta >  30.0f) theta =  30.0f;
  if (theta < -30.0f) theta = -30.0f;

  return theta;
}

// ====================================================================
// PROTÓTIPOS
// ====================================================================
float calcularControle();
float calcularControlePINN();
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

  Serial.println("Sistema Pronto.");
  Serial.println("[A] Ativar | [D] Desativar | [R] Reset | [P] Alternar PINN/Analitico | [C] Coleta");
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
    float u        = usarPINN ? calcularControlePINN() : calcularControle();
    int angulo_final = 120;

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
      Serial.printf("%.3f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        tempoAnteriorMicros / 1e6f,
        state[0],
        state[1],
        setpoint,
        integrator,
        u);
    } else {
      Serial.print(usarPINN ? "[PINN] " : "[ANAL] ");
      Serial.print("Ref:");    Serial.print(setpoint);
      Serial.print(" | Pos:"); Serial.print(state[0]);
      Serial.print(" | U:");   Serial.print(u);
      Serial.print(" | Ang:"); Serial.println(120 - (int)u);
    }
  }
}

// ====================================================================
// FUNÇÕES DE CONTROLE
// ====================================================================

float calcularControle() {
  float erro = setpoint - state[0];

  if (controladorAtivo) {
    integrator += erro * Ts;
  }

  float u_K   = -K[0] * state[0] - K[1] * state[1];
  float u_int = Ki * integrator;
  float u_out = u_K + u_int;

  if (u_out >  60.0f) u_out =  60.0f;
  if (u_out < -60.0f) u_out = -60.0f;

  return u_out;
}

float calcularControlePINN() {
  if (controladorAtivo) {
    integrator += (setpoint - state[0]) * Ts;
  }

  float theta = pinnInference(state[0], state[1], setpoint, integrator);

  return theta;
}

// ====================================================================
// PROCESSAMENTO DE COMANDOS
// ====================================================================
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
    } else if (cmd == 'P') {
      usarPINN = !usarPINN;
      integrator = 0.0f;   // reseta integrador ao trocar controlador
      Serial.print("\n>>> CONTROLADOR: ");
      Serial.println(usarPINN ? "PINN" : "ANALITICO");
    }
  } else {
    float val = atof(serialBuffer);
    if (val > 0.0f) {
      setpoint = val;
      integrator = 0.0f;   // reseta integrador ao mudar setpoint
      Serial.print("\n>>> NOVA REF: ");
      Serial.println(setpoint);
    }
  }

  bufferIndex = 0;
}