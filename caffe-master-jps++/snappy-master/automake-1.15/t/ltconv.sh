#!/bin/sh
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

# Test for libtool convenience libraries.
# This example is taken from the manual.

required='cc native libtoolize'
. test-init.sh

cat >>configure.ac <<'END'
AC_PROG_CC
AM_PROG_AR
AC_PROG_LIBTOOL
AC_CONFIG_FILES([sub1/Makefile
                 sub2/Makefile
                 sub2/sub21/Makefile
                 sub2/sub22/Makefile])
AC_OUTPUT
END

mkdir sub1
mkdir sub2
mkdir sub2/sub21
mkdir sub2/sub22
mkdir empty

cat >Makefile.am <<'END'
SUBDIRS = sub1 sub2
lib_LTLIBRARIES = libtop.la
libtop_la_SOURCES =
libtop_la_LIBADD = \
  sub1/libsub1.la \
  sub2/libsub2.la

bin_PROGRAMS = ltconvtest
ltconvtest_SOURCES = test.c
ltconvtest_LDADD = libtop.la

check-local:
	./ltconvtest$(EXEEXT)
	: > check-ok
installcheck-local:
	$(bindir)/ltconvtest$(EXEEXT)
	: > installcheck-ok
END

cat >sub1/Makefile.am <<'END'
noinst_LTLIBRARIES = libsub1.la
libsub1_la_SOURCES = sub1.c
END

echo 'int sub1 () { return 1; }' > sub1/sub1.c

cat >sub2/Makefile.am <<'END'
SUBDIRS = sub21 sub22
noinst_LTLIBRARIES = libsub2.la
libsub2_la_SOURCES = sub2.c
libsub2_la_LIBADD = \
  sub21/libsub21.la \
  sub22/libsub22.la
END

echo 'int sub2 () { return 2; }' > sub2/sub2.c

cat >sub2/sub21/Makefile.am <<'END'
noinst_LTLIBRARIES = libsub21.la
libsub21_la_SOURCES = sub21.c
END

echo 'int sub21 () { return 21; }' > sub2/sub21/sub21.c

cat >sub2/sub22/Makefile.am <<'END'
noinst_LTLIBRARIES = libsub22.la
libsub22_la_SOURCES = sub22.c
END

echo 'int sub22 () { return 22; }' > sub2/sub22/sub22.c

cat >test.c <<'EOF'
#include <stdio.h>
int main ()
{
  if (1 != sub1 ())
    return 1;
  if (2 != sub2 ())
    return 2;
  if (21 != sub21 ())
    return 3;
  if (22 != sub22 ())
    return 4;
  return 0;
}
EOF

libtoolize
$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing

cwd=$(pwd) || fatal_ "getting current working directory"

# Install libraries in lib/, programs in bin/, and the rest in empty/.
# (in fact there is no "rest", so as the name imply empty/ is
# expected to remain empty).
./configure --prefix="$cwd/empty" --libdir="$cwd/lib" --bindir="$cwd/bin"

$MAKE
test -f libtop.la
test -f sub1/libsub1.la
test -f sub2/libsub2.la
test -f sub2/sub21/libsub21.la
test -f sub2/sub22/libsub22.la
$MAKE check
test -f check-ok
rm -f check-ok

$MAKE install
test -f lib/libtop.la
$MAKE installcheck
test -f installcheck-ok
rm -f installcheck-ok

find empty -type f -print > empty.lst
test -s empty.lst && { cat empty.lst; exit 1; }

$MAKE clean
test ! -e libtop.la
test ! -e sub1/libsub1.la
test ! -e sub2/libsub2.la
test ! -e sub2/sub21/libsub21.la
test ! -e sub2/sub22/libsub22.la
test ! -e ltconvtest

$MAKE installcheck
test -f installcheck-ok
rm -f installcheck-ok

$MAKE uninstall
for d in lib bin; do
  find $d -type f -print > $d.lst
  test -s $d.lst && { cat $d.lst; exit 1; }
  : For shells with busted 'set -e'.
done

:
