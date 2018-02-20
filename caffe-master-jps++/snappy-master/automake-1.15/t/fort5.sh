#! /bin/sh
# Copyright (C) 2006-2014 Free Software Foundation, Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Test mixing Fortran 77 and Fortran (FC), libtool version.

# For now, require the GNU compilers (to avoid some Libtool/Autoconf
# issues).
required='libtoolize g77 gfortran'
. test-init.sh

mkdir sub

cat >hello.f <<'END'
      program hello
      call foo
      call bar
      call goodbye
      stop
      end
END

cat >bye.f90 <<'END'
subroutine goodbye
  call baz
  return
end
END

cat >foo.f90 <<'END'
      subroutine foo
      return
      end
END

sed s,foo,bar, foo.f90 > sub/bar.f90
sed s,foo,baz, foo.f90 > sub/baz.f

cat >>configure.ac <<'END'
AC_PROG_F77
AC_PROG_FC
AC_FC_SRCEXT([f90], [],
  [AC_MSG_FAILURE([$FC compiler cannot create executables], 77)])
AC_FC_LIBRARY_LDFLAGS
AM_PROG_AR
LT_PREREQ([2.0])
AC_PROG_LIBTOOL
AC_OUTPUT
END

cat >Makefile.am <<'END'
bin_PROGRAMS = hello
lib_LTLIBRARIES = libhello.la
noinst_LTLIBRARIES = libgoodbye.la
hello_SOURCES = hello.f
hello_LDADD = libhello.la
libhello_la_SOURCES = foo.f90 sub/bar.f90
libhello_la_LIBADD = libgoodbye.la
libgoodbye_la_SOURCES = bye.f90 sub/baz.f
libgoodbye_la_FCFLAGS =
LDADD = $(FCLIBS)
END

libtoolize --force
$ACLOCAL
# FIXME: stop disabling the warnings in the 'unsupported' category
# FIXME: once the 'subdir-objects' option has been mandatory.
$AUTOMAKE -a -Wno-unsupported
$AUTOCONF

# This test requires Libtool >= 2.0.  Earlier Libtool does not
# have the LT_PREREQ macro to cause autoconf failure.
grep LT_PREREQ configure && skip_ "libtool is too old (probably < 2.0)"

# Ensure we use --tag for f90, too.
grep " --tag=FC" Makefile.in

# ./configure may exit with status 77 if no compiler is found,
# or if the compiler cannot compile Fortran 90 files).
./configure
$MAKE
subobjs=$(echo sub/*.lo)
test "$subobjs" = 'sub/*.lo'
$MAKE distcheck

# The following will be fixed in a later patch:
$MAKE distclean
echo 'AUTOMAKE_OPTIONS = subdir-objects' >> Makefile.am
$AUTOMAKE -a
./configure
$MAKE
test ! -e bar.lo
test ! -e baz.lo
test ! -e libgoodbye_la-baz.lo
$MAKE distcheck

:
