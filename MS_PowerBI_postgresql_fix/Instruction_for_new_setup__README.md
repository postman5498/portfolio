##This is an instruction how to fix connectivity issue with MS PowerBI to a Postgresql database

+ The first error message that might appear is because of the Npgsql version.
+ First install the Npgsql version 3.1.8 (NOT the current one!), also in this directory.

+ Once the first error message with the missing components is gone, the missing certificate error will show up.
************
Download the current certificate for AWS from here:
https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_PostgreSQL.html#PostgreSQL.Concepts.General.SSL

+ Convert the .pem file to any certificate file using a converter like a website or openssl:
+ Import the certificate (Win+"R", then "mmc", add snap-in, and add the .crt certificate to trusted root certificates.)

************
The second post in this one explains in more detail:
https://stackoverflow.com/questions/33249814/npgsql-3-0-3-error-with-power-bi-desktop#
