#pragma once

#include <NvInferRuntimeBase.h> // for nvinfer1::ILogger
#include <cstdio> // for printf
#include <string> // for std::string

class DebugLogger : public nvinfer1::ILogger{
    public:
        DebugLogger(
                std::string flag_name = "UNKNOW", 
                Severity severity = Severity::kINFO) : m_flag_name(flag_name), m_severity(severity), nvinfer1::ILogger(){}

        void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept {
            if (severity <= m_severity) {
                switch (severity) {
                    case Severity::kERROR:
                        printf("[%s][ERROR]: %s\n", m_flag_name.c_str(), msg);
                        break;
                    case Severity::kINTERNAL_ERROR:
                        printf("[%s][INTERNAL_ERROR]: %s\n",m_flag_name.c_str(), msg);
                        break;
                    case Severity::kINFO:
                        printf("[%s][INFO]: %s\n", m_flag_name.c_str(), msg);
                        break;
                    case Severity::kVERBOSE:
                        printf("[%s][VERBOSE]: %s\n", m_flag_name.c_str(), msg);
                        break;
                    case Severity::kWARNING:
                        printf("[%s][WARNING]: %s\n", m_flag_name.c_str(), msg);
                        break;
                }
                
            }
        }
    private:
        std::string m_flag_name;
        Severity m_severity;
};
