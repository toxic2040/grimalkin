# Guardian provenance

The guardian was developed in a standalone Rust repository (`Familiar`) and
folded into grimalkin as a squashed import, so this repo's history starts from
a clean, MIT-licensed tree. The full development history — the deterministic
core, the Linux body, the red-team hardening pass, and the control deck — is
preserved in that original local repository and is not reproduced here.

The guardian was relicensed from AGPL-3.0-or-later to MIT and had its only
copyleft dependency (the GPL-3.0 `rustables` netlink crate) removed in favor of
the `nft` userspace binary before this import.
