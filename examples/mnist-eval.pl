#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Std;

my %opts = ();
getopts('p', \%opts);

my @cnt = (0, 0);
while (<>) {
	next if /^#/;
	chomp;
	my @t = split("\t");
	my @s = split(":", $t[0]);
	my $t = $s[1];
	my ($max, $max_i) = (-1, -1);
	++$cnt[0];
	for (my $i = 1; $i < @t; ++$i) {
		($max, $max_i) = ($t[$i], $i-1) if $max < $t[$i];
	}
	++$cnt[1] if $max_i != $t;
	if (defined($opts{p}) && $max_i != $t) {
		print "$_\n";
	}
}
printf("Error rate: %.2f%%\n", 100.*$cnt[1]/$cnt[0]);
