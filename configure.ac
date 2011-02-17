#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([rnn], [1.0], [jnamika@gmail.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_CONFIG_SRCDIR([src/common/rnn.c])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

# Checks for programs.
AC_PROG_CC_C99
AC_OPENMP
AC_PROG_LIBTOOL

# Checks for libraries.
AC_CHECK_LIB([m], [main])

# Checks for header files.
AC_CHECK_HEADERS([float.h limits.h stddef.h stdint.h stdlib.h string.h unistd.h])
AC_HEADER_ASSERT

AC_ARG_ENABLE([mtrace],
[  --enable-mtrace         whether to profile memory usage with mtrace/muntrace],
[\
case "${enableval}" in
    yes) enable_mtrace=yes ;;
    no)  enable_mtrace=no ;;
    *)   AC_MSG_ERROR([bad value for --enable-mtrace]) ;;
esac],
[enable_mtrace=no])
if test x"${enable_mtrace}" = x"yes"; then
    AC_CHECK_HEADER([mcheck.h], [AC_DEFINE([ENABLE_MTRACE], [1], [Define 1 if you want to profile memory usage with mtrace/muntrace])])
fi

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_C_RESTRICT
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT32_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([atexit memmove memset pow sqrt strchr strpbrk strstr strtoul])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/python/Makefile
                 src/rnn-generate/Makefile
                 src/rnn-learn/Makefile
                 src/rnn-lyapunov/Makefile
                 src/unit-test/Makefile])
AC_OUTPUT
