#! /bin/sh
# Copyright (C) 2008-2014 Free Software Foundation, Inc.
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

# Check all notrans_, dist_, nodist_ prefix combinations for MANS
# primary and install-man dependencies.

. test-init.sh

cat >>configure.ac <<'END'
AC_OUTPUT
END

cat > Makefile.am << 'EOF'
man_MANS = foo.1 foo2.1
dist_man_MANS = bar.2 bar2.2
nodist_man_MANS = baz.3 baz2.3
notrans_man_MANS = x-foo.4 x-foo2.4
notrans_dist_man_MANS = x-bar.5 x-bar2.5
notrans_nodist_man_MANS = x-baz.6 x-baz2.6
man7_MANS = y-foo.man y-foo2.man
dist_man5_MANS = y-bar.man y-bar2.man
nodist_man4_MANS = y-baz.man y-baz2.man
notrans_man3_MANS = z-foo.man z-foo2.man
notrans_dist_man2_MANS = z-bar.man z-bar2.man
notrans_nodist_man1_MANS = z-baz.man z-baz2.man

# These two are ignored.
dist_notrans_man_MANS = nosuch.8
nodist_notrans_man9_MANS = nosuch.man

y-foo.man y-foo2.man:
	: >$@
y-bar.man y-bar2.man:
	: >$@
y-baz.man y-baz2.man:
	: >$@
z-foo.man z-foo2.man:
	: >$@
z-bar.man z-bar2.man:
	: >$@
z-baz.man z-baz2.man:
	: >$@

test-install: install
	test -f inst/man/man1/gnu-foo.1
	test -f inst/man/man1/gnu-foo2.1
	test -f inst/man/man2/gnu-bar.2
	test -f inst/man/man2/gnu-bar2.2
	test -f inst/man/man3/gnu-baz.3
	test -f inst/man/man3/gnu-baz2.3
	test -f inst/man/man4/x-foo.4
	test -f inst/man/man4/x-foo2.4
	test -f inst/man/man5/x-bar.5
	test -f inst/man/man5/x-bar2.5
	test -f inst/man/man6/x-baz.6
	test -f inst/man/man6/x-baz2.6
	test -f inst/man/man7/gnu-y-foo.7
	test -f inst/man/man7/gnu-y-foo2.7
	test -f inst/man/man5/gnu-y-bar.5
	test -f inst/man/man5/gnu-y-bar2.5
	test -f inst/man/man4/gnu-y-baz.4
	test -f inst/man/man4/gnu-y-baz2.4
	test -f inst/man/man3/z-foo.3
	test -f inst/man/man3/z-foo2.3
	test -f inst/man/man2/z-bar.2
	test -f inst/man/man2/z-bar2.2
	test -f inst/man/man1/z-baz.1
	test -f inst/man/man1/z-baz2.1
	test ! -d inst/man/man8
	test ! -d inst/man/man9
EOF

: > foo.1
: > foo2.1
: > bar.2
: > bar2.2
: > baz.3
: > baz2.3
: > x-foo.4
: > x-foo2.4
: > x-bar.5
: > x-bar2.5
: > x-baz.6
: > x-baz2.6

$ACLOCAL
$AUTOCONF
$AUTOMAKE

grep '^install-man1:' Makefile.in | grep '\$(man_MANS)'
grep '^install-man2:' Makefile.in | grep '\$(dist_man_MANS)'
grep '^install-man3:' Makefile.in | grep '\$(nodist_man_MANS)'
grep '^install-man4:' Makefile.in | grep '\$(notrans_man_MANS)'
grep '^install-man5:' Makefile.in | grep '\$(notrans_dist_man_MANS)'
grep '^install-man6:' Makefile.in | grep '\$(notrans_nodist_man_MANS)'
grep '^install-man8:' Makefile.in && exit 1
grep '^install-man9:' Makefile.in && exit 1

cwd=$(pwd) || fatal_ "getting current working directory"

./configure --program-prefix=gnu- --prefix "$cwd"/inst \
                                  --mandir "$cwd"/inst/man
$MAKE
$MAKE test-install
test $(find inst/man -type f -print | wc -l) -eq 24
$MAKE uninstall
test $(find inst/man -type f -print | wc -l) -eq 0

# Opportunistically test for installdirs.
rm -rf inst
$MAKE installdirs
test -d inst/man/man1
test -d inst/man/man2
test -d inst/man/man3
test -d inst/man/man4
test -d inst/man/man5
test -d inst/man/man6
test -d inst/man/man7
test -d inst/man/man8 && exit 1
test -d inst/man/man9 && exit 1

:
