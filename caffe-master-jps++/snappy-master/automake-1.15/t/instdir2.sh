#! /bin/sh
# Copyright (C) 2009-2014 Free Software Foundation, Inc.
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

# If $(foodir) is the empty string, then nothing should be installed there.
# This test ensures this also if $(foo_PRIMARY) is nonempty, see
# 'instdir.sh'.

. test-init.sh

cat >>configure.ac <<'END'
AC_SUBST([foodir], ['${datadir}'/foo])
AC_OUTPUT
END

mkdir sub

cat >Makefile.am <<'END'
bin_SCRIPTS = s
nobase_bin_SCRIPTS = ns sub/ns
data_DATA = d
nobase_data_DATA = nd sub/nd
include_HEADERS = h
nobase_include_HEADERS = nh sub/nh
foo_DATA = f
nobase_foo_DATA = nf sub/nf
bardir = $(datadir)/bar
bar_DATA = b
nobase_bar_DATA = nb sub/nb
man1_MANS = m1.1
man_MANS = m.2
notrans_man1_MANS = nm1.1
notrans_man_MANS = nm.2
END

: >s
: >ns
: >sub/ns
: >d
: >nd
: >sub/nd
: >h
: >nh
: >sub/nh
: >f
: >nf
: >sub/nf
: >b
: >nb
: >sub/nb
: >m1.1
: >m.2
: >nm1.1
: >nm.2

$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing

cwd=$(pwd) || fatal_ "getting current working directory"
instdir=$cwd/inst
destdir=$cwd/dest
mkdir build
cd build
../configure --prefix="$instdir"
$MAKE

nulldirs='bindir= datadir= includedir= foodir= bardir= man1dir= man2dir='
null_install

:
