#! /bin/sh
# Copyright (C) 2001-2014 Free Software Foundation, Inc.
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

# Test installation with substitutions.  This test is based on
# 'nobase.sh'.

required=cc
. test-init.sh

cat >> configure.ac <<'EOF'
AC_PROG_CC
AM_PROG_AR
AC_PROG_RANLIB
if test x"$doit" = x"yes"; then
  AC_SUBST([basehdr], [sub/base.h])
  AC_SUBST([nobasehdr], [sub/nobase.h])
  AC_SUBST([basedata], [sub/base.dat])
  AC_SUBST([nobasedata], [sub/nobase.dat])
  AC_SUBST([basescript], [sub/base.sh])
  AC_SUBST([nobasescript], [sub/nobase.sh])
  AC_SUBST([baseprog], ['sub/base$(EXEEXT)'])
  AC_SUBST([nobaseprog], ['sub/nobase$(EXEEXT)'])
  AC_SUBST([baselib], [sub/libbase.a])
  AC_SUBST([nobaselib], [sub/libnobase.a])
fi
AC_OUTPUT
EOF

cat > Makefile.am << 'EOF'
foodir = $(prefix)/foo
fooexecdir = $(prefix)/foo

foo_HEADERS = @basehdr@
nobase_foo_HEADERS = @nobasehdr@
EXTRA_HEADERS = sub/base.h sub/nobase.h

dist_foo_DATA = @basedata@
nobase_dist_foo_DATA = @nobasedata@

dist_fooexec_SCRIPTS = @basescript@
nobase_dist_fooexec_SCRIPTS = @nobasescript@
EXTRA_SCRIPTS = sub/base.sh sub/nobase.sh

fooexec_PROGRAMS = @baseprog@
nobase_fooexec_PROGRAMS = @nobaseprog@
EXTRA_PROGRAMS = sub/base sub/nobase
sub_base_SOURCES = source.c
sub_nobase_SOURCES = source.c

fooexec_LIBRARIES = @baselib@
nobase_fooexec_LIBRARIES = @nobaselib@
EXTRA_LIBRARIES = sub/libbase.a sub/libnobase.a
sub_libbase_a_SOURCES = source.c
sub_libnobase_a_SOURCES = source.c

test-install-data: install-data
	test   -f inst/foo/sub/nobase.h
	test ! -f inst/foo/nobase.h
	test   -f inst/foo/base.h
	test   -f inst/foo/sub/nobase.dat
	test ! -f inst/foo/nobase.dat
	test   -f inst/foo/base.dat
	test ! -f inst/foo/sub/pnobase.sh
	test ! -f inst/foo/pbase.sh
	test ! -f inst/foo/sub/pnobase$(EXEEXT)
	test ! -f inst/foo/pbase$(EXEEXT)
	test ! -f inst/foo/sub/libnobase.a
	test ! -f inst/foo/libbase.a

test-install-exec: install-exec
	test   -f inst/foo/sub/pnobase.sh
	test ! -f inst/foo/pnobase.sh
	test   -f inst/foo/pbase.sh
	test   -f inst/foo/sub/pnobase$(EXEEXT)
	test ! -f inst/foo/pnobase$(EXEEXT)
	test   -f inst/foo/pbase$(EXEEXT)
	test   -f inst/foo/sub/libnobase.a
	test ! -f inst/foo/libnobase.a
	test   -f inst/foo/libbase.a

test-install-nothing-data: install-data
	test ! -f inst/foo/sub/nobase.h
	test ! -f inst/foo/nobase.h
	test ! -f inst/foo/base.h
	test ! -f inst/foo/sub/nobase.dat
	test ! -f inst/foo/nobase.dat
	test ! -f inst/foo/base.dat
	test ! -f inst/foo/sub/pnobase.sh
	test ! -f inst/foo/pbase.sh
	test ! -f inst/foo/sub/pnobase$(EXEEXT)
	test ! -f inst/foo/pbase$(EXEEXT)
	test ! -f inst/foo/sub/libnobase.a
	test ! -f inst/foo/libbase.a

test-install-nothing-exec: install-exec
	test ! -f inst/foo/sub/pnobase.sh
	test ! -f inst/foo/pnobase.sh
	test ! -f inst/foo/pbase.sh
	test ! -f inst/foo/sub/pnobase$(EXEEXT)
	test ! -f inst/foo/pnobase$(EXEEXT)
	test ! -f inst/foo/pbase$(EXEEXT)
	test ! -f inst/foo/sub/libnobase.a
	test ! -f inst/foo/libnobase.a
	test ! -f inst/foo/libbase.a
EOF

mkdir sub

: > sub/base.h
: > sub/nobase.h
: > sub/base.dat
: > sub/nobase.dat
: > sub/base.sh
: > sub/nobase.sh

cat >source.c <<'EOF'
int
main (int argc, char *argv[])
{
  return 0;
}
EOF
cp source.c source2.c

rm -f install-sh

$ACLOCAL
$AUTOCONF
$AUTOMAKE -a --copy
./configure --prefix "$(pwd)/inst" --program-prefix=p doit=yes

$MAKE
$MAKE test-install-data
$MAKE test-install-exec
$MAKE uninstall
$MAKE clean

test $(find inst/foo -type f -print | wc -l) -eq 0

./configure --prefix "$(pwd)/inst" --program-prefix=p doit=no

$MAKE
$MAKE test-install-nothing-data
$MAKE test-install-nothing-exec
$MAKE uninstall


# Likewise, in a VPATH build.

$MAKE distclean
mkdir build
cd build
../configure --prefix "$(pwd)/inst" --program-prefix=p doit=yes
$MAKE
$MAKE test-install-data
$MAKE test-install-exec
$MAKE uninstall
test $(find inst/foo -type f -print | wc -l) -eq 0

../configure --prefix "$(pwd)/inst" --program-prefix=p doit=no
$MAKE
$MAKE test-install-nothing-data
$MAKE test-install-nothing-exec

:
