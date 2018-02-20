#! /bin/sh
# Copyright (C) 2003-2014 Free Software Foundation, Inc.
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

# Check multilib support.
# Based on a test case from Ralf Corsepius.

required='gcc GNUmake'
. test-init.sh

mldir=$am_top_srcdir/contrib/multilib
mkdir m4
cp "$mldir"/config-ml.in "$mldir"/symlink-tree .
cp "$mldir"/multi.m4 m4

ACLOCAL_PATH=${ACLOCAL_PATH+"$ACLOCAL_PATH:"}$(pwd)/m4
export ACLOCAL_PATH

cat >configure.ac <<'END'
AC_INIT([multlib], [1.0])
AC_CONFIG_SRCDIR(libfoo/foo.c)
AC_CONFIG_AUX_DIR(.)
AM_INIT_AUTOMAKE
AC_CONFIG_FILES([Makefile])
AC_CONFIG_SUBDIRS(libfoo)
AC_CONFIG_SUBDIRS(libbar)
AC_OUTPUT
END

cat >mycc <<'END'
#! /bin/sh
case ${1+"$@"} in
 *-print-multi-lib*)
  echo ".;"
  echo "debug;@g"
  exit 0 ;;
esac
gcc ${1+"$@"}
END

chmod +x mycc
PATH=$(pwd)$PATH_SEPARATOR$PATH; export PATH

cat >Makefile.am <<'EOF'
SUBDIRS = @subdirs@
EXTRA_DIST = config-ml.in symlink-tree
check-all:
	test -f debug/libfoo/libfoo.a
	test -f debug/libbar/libbar.a
	test -f libfoo/libfoo.a
	test -f libbar/libbar.a
EOF

# libfoo tests multilib supports when there are no subdirectories
# libbar tests multilib supports when there are subdirectories

mkdir libfoo
cp "$mldir"/multilib.am libfoo/

cat >libfoo/configure.ac <<'END'
AC_PREREQ(2.57)
AC_INIT(libfoo, 0.1, nobody@localhost)
AC_CONFIG_SRCDIR(foo.c)
# Apparently it doesn't work to have auxdir=.. when
# multilib uses symlinked trees.
AC_CONFIG_AUX_DIR(.)
AM_INIT_AUTOMAKE
AC_PROG_CC
AM_PROG_AR
AC_PROG_RANLIB
AM_ENABLE_MULTILIB(Makefile,[..])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
END

cat >libfoo/Makefile.am <<'END'
noinst_LIBRARIES = libfoo.a
libfoo_a_SOURCES = foo.c
include $(top_srcdir)/multilib.am
END

: > libfoo/foo.c

mkdir libbar
cp "$mldir"/multilib.am libbar/

cat >libbar/configure.ac <<'END'
AC_PREREQ(2.57)
AC_INIT(libbar, 0.1, nobody@localhost)
# Apparently it doesn't work to have auxdir=.. when
# multilib uses symlinked trees.
AC_CONFIG_AUX_DIR(.)
AM_INIT_AUTOMAKE
AC_PROG_CC
AM_PROG_AR
AC_PROG_RANLIB
AM_ENABLE_MULTILIB(Makefile,[..])
AC_CONFIG_FILES([Makefile sub/Makefile])
AC_OUTPUT
END

cat >libbar/Makefile.am <<'END'
SUBDIRS = sub
noinst_LIBRARIES = libbar.a
libbar_a_SOURCES = bar.c
include $(top_srcdir)/multilib.am
END

mkdir libbar/sub
echo 'include $(top_srcdir)/multilib.am' >libbar/sub/Makefile.am
: > libbar/bar.c

$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing

cd libfoo
$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing
cd ..

cd libbar
$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing
cd ..

# Check VPATH builds
mkdir build
cd build
../configure --enable-multilib CC=mycc
$MAKE
test -f debug/libfoo/libfoo.a
test -f debug/libbar/libbar.a
test -f libfoo/libfoo.a
test -f libbar/libbar.a
$MAKE install
$MAKE distcleancheck

# Check standard builds.
cd ..
# Why to I have to specify --with-target-subdir?
./configure --enable-multilib --with-target-subdir=. CC=mycc
$MAKE check
DISTCHECK_CONFIGURE_FLAGS='--enable-multilib CC=mycc' $MAKE distcheck

:
