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

# Make sure nobase_* works.

required=cc
. test-init.sh

cat >> configure.ac <<'EOF'
AC_PROG_CC
AM_PROG_AR
AC_PROG_RANLIB
AC_OUTPUT
EOF

cat > Makefile.am << 'EOF'
foodir = $(prefix)/foo
fooexecdir = $(prefix)/foo

foo_HEADERS = sub/base.h sub/base-gen.h
nobase_foo_HEADERS = sub/nobase.h sub/nobase-gen.h

dist_foo_DATA = sub/base.dat sub/base-gen.dat
nobase_dist_foo_DATA = sub/nobase.dat sub/nobase-gen.dat

dist_fooexec_SCRIPTS = sub/base.sh sub/base-gen.sh
nobase_dist_fooexec_SCRIPTS = sub/nobase.sh sub/nobase-gen.sh

fooexec_PROGRAMS = sub/base
nobase_fooexec_PROGRAMS = sub/nobase
sub_base_SOURCES = source.c
sub_nobase_SOURCES = source.c

fooexec_LIBRARIES = sub/libbase.a
nobase_fooexec_LIBRARIES = sub/libnobase.a
sub_libbase_a_SOURCES = source.c
sub_libnobase_a_SOURCES = source.c

generated_files = sub/base-gen.h sub/nobase-gen.h sub/base-gen.dat \
sub/nobase-gen.dat sub/base-gen.sh sub/nobase-gen.sh

$(generated_files):
	$(MKDIR_P) sub
	echo "generated file $@" > $@

CLEANFILES = $(generated_files)

test-install-data: install-data
	test   -f inst/foo/sub/nobase.h
	test ! -f inst/foo/nobase.h
	test   -f inst/foo/sub/nobase-gen.h
	test ! -f inst/foo/nobase-gen.h
	test   -f inst/foo/base.h
	test   -f inst/foo/base-gen.h
	test   -f inst/foo/sub/nobase.dat
	test ! -f inst/foo/nobase.dat
	test   -f inst/foo/sub/nobase-gen.dat
	test ! -f inst/foo/nobase-gen.dat
	test   -f inst/foo/base.dat
	test   -f inst/foo/base-gen.dat
	test ! -f inst/foo/sub/pnobase.sh
	test ! -f inst/foo/sub/pnobase-gen.sh
	test ! -f inst/foo/pbase.sh
	test ! -f inst/foo/pbase-gen.sh
	test ! -f inst/foo/sub/pnobase$(EXEEXT)
	test ! -f inst/foo/pbase$(EXEEXT)
	test ! -f inst/foo/sub/libnobase.a
	test ! -f inst/foo/libbase.a

test-install-exec: install-exec
	test   -f inst/foo/sub/pnobase.sh
	test ! -f inst/foo/pnobase.sh
	test   -f inst/foo/sub/pnobase-gen.sh
	test ! -f inst/foo/pnobase-gen.sh
	test   -f inst/foo/pbase.sh
	test   -f inst/foo/pbase-gen.sh
	test   -f inst/foo/sub/pnobase$(EXEEXT)
	test ! -f inst/foo/pnobase$(EXEEXT)
	test   -f inst/foo/pbase$(EXEEXT)
	test   -f inst/foo/sub/libnobase.a
	test ! -f inst/foo/libnobase.a
	test   -f inst/foo/libbase.a

.PHONY: test-install-exec test-install-data
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
./configure --prefix "$(pwd)/inst" --program-prefix=p

$MAKE
$MAKE test-install-data
$MAKE test-install-exec
$MAKE uninstall

test $(find inst/foo -type f -print | wc -l) -eq 0

$MAKE install-strip

# Likewise, in a VPATH build.

$MAKE uninstall
$MAKE distclean
mkdir build
cd build
../configure --prefix "$(pwd)/inst" --program-prefix=p
$MAKE
$MAKE test-install-data
$MAKE test-install-exec
$MAKE uninstall
test $(find inst/foo -type f -print | wc -l) -eq 0

:
