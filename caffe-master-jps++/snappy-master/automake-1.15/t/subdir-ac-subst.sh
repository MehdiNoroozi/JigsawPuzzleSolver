#! /bin/sh
# Copyright (C) 2002-2014 Free Software Foundation, Inc.
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

# The for conditional SUBDIRS.
# SUBDIRS + AC_SUBST setup from the manual.
# Lots of lines here are duplicated in 'subcond-am-cond.sh'.

. test-init.sh

cat >> configure.ac <<'END'
if test "$want_opt" = yes; then
  MAYBE_OPT=opt
else
  MAYBE_OPT=
fi
AC_SUBST([MAYBE_OPT])
AC_CONFIG_FILES([src/Makefile opt/Makefile])
AC_OUTPUT
END

cat > Makefile.am <<'END'
SUBDIRS = src $(MAYBE_OPT)
DIST_SUBDIRS = src opt

# Testing targets.
#
# We want to ensure that
#      - src/source and opt/source are always distributed.
#      - src/result is always built
#      - opt/result is built conditionally
#
# We rely on 'distcheck' to run 'check-local' and use
# 'sanity1' and 'sanity2' as evidences that test-build was run.

test_rootdir = $(top_builddir)/../../..

test-build: all
	test -f src/result
	if test -n "$(MAYBE_OPT)"; then \
	   test -f opt/result || exit 1; \
	   : > $(test_rootdir)/sanity2 || exit 1; \
	else \
	   test ! -f opt/result || exit 1; \
	   : > $(test_rootdir)/sanity1 || exit 1; \
	fi

test-dist: distdir
	test -f $(distdir)/src/source
	test -f $(distdir)/opt/source

check-local: test-build test-dist
END

mkdir src opt
: > src/source
: > opt/source

cat > src/Makefile.am <<'END'
EXTRA_DIST = source
all-local: result
CLEANFILES = result

result: source
	cp $(srcdir)/source result
END

# We want in opt/ the same Makefile as in src/.  Let's exercise 'include'.
cat > opt/Makefile.am <<'END'
include ../src/Makefile.am
END

$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing
./configure
$MAKE distcheck
test -f sanity1
DISTCHECK_CONFIGURE_FLAGS=want_opt=yes $MAKE distcheck
test -f sanity2

:
