"""Mixture definitions for zest."""

import t5

from . import tasks

# N.B. Tasks must be imported before mixtures, so that mixtures can use tasks
# in their definitions.


t5.data.MixtureRegistry.add("zest", ["zest"], default_rate=1.0)

t5.data.MixtureRegistry.add(
    "zest_glue",
    [
        "zest",
        "glue_cola_v002",
        "glue_sst2_v002",
        "glue_mrpc_v002",
        "glue_stsb_v002",
        "glue_qqp_v002",
        "glue_mnli_v002",
        "glue_qnli_v002",
        "glue_rte_v002",
        "glue_wnli_v002",
    ],
    default_rate=1.0,
)

t5.data.MixtureRegistry.add(
    "zest_super_glue",
    [
        "zest",
        "super_glue_boolq_v102",
        "super_glue_cb_v102",
        "super_glue_copa_v102",
        "super_glue_multirc_v102",
        "super_glue_record_v102",
        "super_glue_rte_v102",
        "super_glue_wic_v102",
    ],
    default_rate=1.0,
)

t5.data.MixtureRegistry.add(
    "zest_qa",
    [
        "zest",
        "super_glue_boolq_v102",
        "super_glue_multirc_v102",
        "super_glue_record_v102",
        "squad_v010_allanswers",
    ],
    default_rate=1.0,
)
