#! /bin/sh
# Copyright (C) 2005-2014 Free Software Foundation, Inc.
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

# Check support for AC_CONFIG_LIBOBJ_DIR vs LIBOBJS.
# (pr401b.sh and pr401c.sh do the same for LTLIBOBJS and ALLOCA)

required=cc
. test-init.sh

mkdir lib src

cat >lib/feep.c <<'EOF'
const char *feep (void)
{
  return "feep";
}
EOF

cat >src/main.c <<'EOF'
#include <stdio.h>

extern const char *feep (void);

int main (void)
{
  puts (feep ());
  return 0;
}
EOF

cat >>configure.ac << 'EOF'
## These lines are activated for later tests
#: AC_CONFIG_LIBOBJ_DIR([lib])
AC_PROG_CC
AC_LIBOBJ([feep])
AC_LIBSOURCE([feep.c])
AM_PROG_AR
AC_PROG_RANLIB
AC_CONFIG_FILES([lib/Makefile src/Makefile])
AM_CONDITIONAL([CROSS_COMPILING], [test $cross_compiling = yes])
AC_OUTPUT
EOF

## ------------------------------------------ ##
## First a test of traditional LIBOBJS usage. ##
## ------------------------------------------ ##

cat >Makefile.am <<'EOF'
SUBDIRS = lib src
EOF

cat >lib/Makefile.am <<'EOF'
noinst_LIBRARIES = libfeep.a
libfeep_a_SOURCES =
libfeep_a_LIBADD = $(LIBOBJS)
EOF

cat >src/Makefile.am <<'EOF'
check_PROGRAMS = main
main_LDADD = ../lib/libfeep.a

if !CROSS_COMPILING
TESTS = main
endif
EOF

cp "$am_scriptdir/ar-lib" . || fatal_ "fetching auxiliary script 'ar-lib'"

$ACLOCAL
$AUTOCONF
$AUTOMAKE -a
./configure
$MAKE distcheck

## ------------------------------------------ ##
## Traditional LIBOBJS with LIBOBJDIR set.    ##
## ------------------------------------------ ##

# Invocation of AC_CONFIG_LIBOBJ_DIR may be necessary for reasons
# unrelated to Automake or Makefile.am layout.

sed 's/#: //' configure.ac >configure.tmp
mv -f configure.tmp configure.ac

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure
test ! -e lib/lib
$MAKE distcheck

## -------------------------------------------- ##
## Error message with usage in wrong directory. ##
## -------------------------------------------- ##

mv -f src/Makefile.am src/t
sed 's/LDADD = .*/LDADD = @LIBOBJS@/' src/t > src/Makefile.am
AUTOMAKE_fails
grep 'cannot be used outside.*lib' stderr
mv -f src/t src/Makefile.am


## -------------------------------------------- ##
## Test using LIBOBJS from a sibling directory. ##
## -------------------------------------------- ##

sed 's/lib\/Makefile //' configure.ac >configure.tmp
mv -f configure.tmp configure.ac

cat >Makefile.am <<'EOF'
SUBDIRS = src
EOF

cat > src/Makefile.am <<'EOF'
AUTOMAKE_OPTIONS = subdir-objects

noinst_LIBRARIES = libfeep.a
libfeep_a_SOURCES =
libfeep_a_LIBADD = $(LIBOBJS)

check_PROGRAMS = main
main_LDADD = libfeep.a

if !CROSS_COMPILING
TESTS = main
endif
EOF

$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing
./configure
test ! -e src/lib
test ! -e 'src/$(top_builddir)'
$MAKE
$MAKE check
$MAKE distclean


## ----------------------------------------- ##
## Test using LIBOBJS from parent directory. ##
## ----------------------------------------- ##

sed 's/^.*src\/Makefile.*$//' configure.ac >configure.tmp
mv -f configure.tmp configure.ac

cat >Makefile.am <<'EOF'
AUTOMAKE_OPTIONS = subdir-objects

noinst_LIBRARIES = lib/libfeep.a
lib_libfeep_a_SOURCES =
lib_libfeep_a_LIBADD = $(LIBOBJS)

check_PROGRAMS = src/main
src_main_SOURCES = src/main.c
src_main_LDADD = lib/libfeep.a

if !CROSS_COMPILING
TESTS = src/main
endif

check-local:
	test -f src/main.$(OBJEXT)
	test -f lib/feep.$(OBJEXT)
	test ! -f src/$(DEPDIR)/feep.Po
EOF

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure
$MAKE distcheck

:
