# tpch

## About

TPC-H benchmark data, cut down to its two fixed tables: region has one row per world region and nation one row per country, each nation in one region. Every comment column is random filler text.

## Tables

### region (5 rows)

- `r_regionkey` BIGINT: Region number, 0 to 4.
- `r_name` VARCHAR: Region name, in upper case.
  Values: 'AFRICA', 'AMERICA', 'ASIA', 'EUROPE', 'MIDDLE EAST'
- `r_comment` VARCHAR: Random filler text.
  Values: 'ges. thinly even pinto beans ca', 'hs use ironic, even requests. s', 'lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to ', 'ly final courts cajole furiously final excuse', 'uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl'

### nation (25 rows)

- `n_nationkey` BIGINT: Nation number, 0 to 24.
- `n_name` VARCHAR: Nation (country) name, in upper case.
  Values: 'ALGERIA', 'ARGENTINA', 'BRAZIL', 'CANADA', 'CHINA', 'EGYPT', 'ETHIOPIA', 'FRANCE', 'GERMANY', 'INDIA', 'INDONESIA', 'IRAN', 'IRAQ', 'JAPAN', 'JORDAN', 'KENYA', 'MOROCCO', 'MOZAMBIQUE', 'PERU', 'ROMANIA', 'RUSSIA', 'SAUDI ARABIA', 'UNITED KINGDOM', 'UNITED STATES', 'VIETNAM'
- `n_regionkey` BIGINT: Region the nation is in (region.r_regionkey).
- `n_comment` VARCHAR: Random filler text.
  Values: ' haggle. carefully final deposits detect slyly agai', ' pending excuses haggle furiously deposits. pending, express pinto beans wake fluffily past t', ' requests against the platelets use never according to the quickly regular pint', ' slyly express asymptotes. regular deposits haggle slyly. carefully ironic hockey players sleep blithely. carefull', 'al foxes promise slyly according to the regular accounts. bold requests alon', 'c dependencies. furiously express notornis sleep slyly regular accounts. ideas sleep. depos', 'eans boost carefully special requests. accounts are. carefull', 'eas hang ironic, silent packages. slyly regular packages are furiously over the tithes. fluffily bold', 'efully alongside of the slyly final dependencies. ', 'hely enticingly express accounts. even, final ', 'ic deposits are blithely about the carefully regular pa', 'l platelets. regular accounts x-ray: unusual, regular acco', 'nic deposits boost atop the quickly final requests? quickly regula', 'ously. final, express gifts cajole a', 'platelets. blithely pending dependencies use fluffily across the even pinto beans. carefully silent accoun', 'refully final requests. regular, ironi', 'rns. blithely bold courts among the closely regular packages use furiously bold platelets?', 's. ironic, unusual asymptotes wake blithely r', 'ss excuses cajole slyly across the packages. deposits print aroun', 'ts. silent requests haggle. closely express packages sleep across the blithely', 'ular asymptotes are about the furious multipliers. express dependencies nag above the ironically ironic account', 'ven packages wake quickly. regu', 'y above the carefully unusual theodolites. final dugouts are quickly across the furiously regular d', 'y alongside of the pending deposits. carefully special packages are about the ironic forges. slyly special ', 'y final packages. slow foxes cajole quickly. quickly silent platelets breach ironic accounts. unusual pinto be'

## Keys

Primary keys (unique per row):

- region: `r_regionkey`
- nation: `n_nationkey`

Foreign keys (each value is a key of the other table):

- nation (`n_regionkey`) to region (`r_regionkey`)

## Missing values

- Nothing: every value in every column is real.
