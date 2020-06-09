/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef HELPER_CUDA_GL_H
#define HELPER_CUDA_GL_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// includes, graphics
#if defined (__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

#ifdef __CUDA_GL_INTEROP_H__
////////////////////////////////////////////////////////////////////////////////
// These are CUDA OpenGL Helper functions

static inline const char* glErrorToString(GLenum err)
{
#define CASE_RETURN_MACRO(arg) case arg: return #arg
    switch(err)
    {
        CASE_RETURN_MACRO(GL_NO_ERROR);
        CASE_RETURN_MACRO(GL_INVALID_ENUM);
        CASE_RETURN_MACRO(GL_INVALID_VALUE);
        CASE_RETURN_MACRO(GL_INVALID_OPERATION);
        CASE_RETURN_MACRO(GL_OUT_OF_MEMORY);
        CASE_RETURN_MACRO(GL_STACK_UNDERFLOW);
        CASE_RETURN_MACRO(GL_STACK_OVERFLOW);
#ifdef GL_INVALID_FRAMEBUFFER_OPERATION
        CASE_RETURN_MACRO(GL_INVALID_FRAMEBUFFER_OPERATION);
#endif
        default: break;
    }
#undef CASE_RETURN_MACRO
    return "*UNKNOWN*";
}

////////////////////////////////////////////////////////////////////////////
//! Check for OpenGL error
//! @return bool if no GL error has been encountered, otherwise 0
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
//! @note The GL error is listed on stderr
//! @note This function should be used via the CHECK_ERROR_GL() macro
////////////////////////////////////////////////////////////////////////////
inline bool
sdkCheckErrorGL(const char *file, const int line)
{
    bool ret_val = true;

    // check for error
    GLenum gl_error = glGetError();

    if (gl_error != GL_NO_ERROR)
    {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        char tmpStr[512];
        // NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
        // when the user double clicks on the error line in the Output pane. Like any compile error.
        sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line, glErrorToString(gl_error));
        fprintf(stderr, "%s", tmpStr);
#endif
        fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
        fprintf(stderr, "%s\n", glErrorToString(gl_error));
        ret_val = false;
    }

    return ret_val;
}

#define SDK_CHECK_ERROR_GL()                                              \
    if( false == sdkCheckErrorGL( __FILE__, __LINE__)) {                  \
        DEVICE_RESET                                                      \
        exit(EXIT_FAILURE);                                               \
    }
#endif

#endif
