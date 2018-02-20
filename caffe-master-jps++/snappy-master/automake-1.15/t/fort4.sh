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

# Test mixing Fortran 77 and Fortran (FC).

# For now, require the GNU compilers (to avoid some Libtool/Autoconf
# issues).
required='g77 gfortran'
. test-init.sh

mkdir sub

cat >hello.f <<'END'
      program hello
      call foo
      call bar
      stop
      end
END

cat >bye.f90 <<'END'
program goodbye
  call baz
  stop
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
AC_OUTPUT
END

cat >Makefile.am <<'END'
bin_PROGRAMS = hello goodbye
hello_SOURCES = hello.f foo.f90 sub/bar.f90
goodbye_SOURCES = bye.f90 sub/baz.f
goodbye_FCFLAGS =
LDADD = $(FCLIBS)
END

$ACLOCAL
$AUTOMAKE -a -Wno-unsupported
# The Fortran 77 linker should be preferred:
grep '.\$(FCLINK)' Makefile.in && exit 1

$AUTOCONF
# ./configure may exit with status 77 if no compiler is found,
# or if the compiler cannot compile Fortran 90 files).
./configure
$MAKE
subobjs=$(echo sub/*.o sub/*.obj)
test "$subobjs" = 'sub/*.o sub/*.obj'
$MAKE distcheck

$MAKE distclean
echo 'AUTOMAKE_OPTIONS = subdir-objects' >> Makefile.am
$AUTOMAKE
./configure
$MAKE
test ! -e bar.o
test ! -e bar.obj
test ! -e baz.o
test ! -e baz.obj
test ! -e goodbye-baz.o
test ! -e goodbye-baz.obj
$MAKE distcheck

:
