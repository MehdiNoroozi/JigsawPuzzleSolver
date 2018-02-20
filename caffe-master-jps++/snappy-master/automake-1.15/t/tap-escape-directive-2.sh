#! /bin/sh
# Copyright (C) 2011-2014 Free Software Foundation, Inc.
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

# TAP support:
#  - "escape" TODO and SKIP directives (by escaping the "#" character)

. test-init.sh

. tap-setup.sh

cat > all.test <<'END'
1..8

not ok \ # TODO
ok \ # SKIP

not ok \\# TODO
ok \\# SKIP

ok \\\# TODO
ok \\\# SKIP

not ok \\\\\\\\\\# TODO
ok     \\\\\\\\\\# SKIP
END

run_make -O check
count_test_results total=8 pass=2 fail=0 xpass=0 xfail=3 skip=3 error=0

grep '^XFAIL: all\.test 1 .*# TODO' stdout
grep '^SKIP: all\.test 2 .*# SKIP' stdout
grep '^XFAIL: all\.test 3 .*# TODO' stdout
grep '^SKIP: all\.test 4 .*# SKIP' stdout
grep '^PASS: all\.test 5 .*# TODO' stdout
grep '^PASS: all\.test 6 .*# SKIP' stdout
grep '^XFAIL: all\.test 7 .*# TODO' stdout
grep '^SKIP: all\.test 8 .*# SKIP' stdout

:
