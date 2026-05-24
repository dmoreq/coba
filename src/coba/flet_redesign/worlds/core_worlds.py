"""Core narrative world configurations for Phase 2."""

from __future__ import annotations

from coba.flet_redesign.worlds.schema import ArmDef, FeatureDef, WorldConfig


RURAL_CLINIC_WORLD = WorldConfig(
    world_id="rural_clinic",
    title="The Rural Clinic",
    description="Select treatment pathways for constrained clinic capacity.",
    difficulty="easy",
    features=(
        FeatureDef(
            name="symptom_severity", feature_type="numeric", numeric_min=0.0, numeric_max=10.0
        ),
        FeatureDef(name="comorbidity", feature_type="binary"),
        FeatureDef(
            name="age_bucket", feature_type="categorical", categories=("young", "adult", "senior")
        ),
    ),
    arms=(
        ArmDef(
            arm_id="standard_care",
            label="Standard Care",
            base_rate=0.58,
            weights={"symptom_severity": -0.2, "comorbidity": -0.1, "age_bucket": -0.05},
        ),
        ArmDef(
            arm_id="targeted_followup",
            label="Targeted Follow-Up",
            base_rate=0.52,
            weights={"symptom_severity": 0.25, "comorbidity": 0.2, "age_bucket": 0.1},
        ),
        ArmDef(
            arm_id="remote_monitoring",
            label="Remote Monitoring",
            base_rate=0.48,
            weights={"symptom_severity": 0.1, "comorbidity": 0.05, "age_bucket": 0.2},
        ),
    ),
)


MOVIEMATCH_WORLD = WorldConfig(
    world_id="moviematch",
    title="MovieMatch",
    description="Personalize a streaming shelf for engagement uplift.",
    difficulty="easy",
    features=(
        FeatureDef(
            name="session_minutes", feature_type="numeric", numeric_min=0.0, numeric_max=180.0
        ),
        FeatureDef(name="new_user", feature_type="binary"),
        FeatureDef(
            name="genre_preference",
            feature_type="categorical",
            categories=("action", "drama", "comedy"),
        ),
    ),
    arms=(
        ArmDef(
            arm_id="trending_now",
            label="Trending Now",
            base_rate=0.5,
            weights={"session_minutes": -0.1, "new_user": 0.2, "genre_preference": 0.05},
        ),
        ArmDef(
            arm_id="personalized_mix",
            label="Personalized Mix",
            base_rate=0.56,
            weights={"session_minutes": 0.15, "new_user": -0.1, "genre_preference": 0.2},
        ),
        ArmDef(
            arm_id="continue_watching",
            label="Continue Watching",
            base_rate=0.54,
            weights={"session_minutes": 0.2, "new_user": -0.2, "genre_preference": -0.05},
        ),
    ),
)


NEWSFEED_WORLD = WorldConfig(
    world_id="newsfeed",
    title="NewsFeed",
    description="Rank article cards to balance relevance and novelty.",
    difficulty="medium",
    features=(
        FeatureDef(name="reading_depth", feature_type="numeric", numeric_min=0.0, numeric_max=1.0),
        FeatureDef(name="breaking_news_mode", feature_type="binary"),
        FeatureDef(
            name="topic_affinity",
            feature_type="categorical",
            categories=("finance", "sports", "tech"),
        ),
    ),
    arms=(
        ArmDef(
            arm_id="breaking_first",
            label="Breaking First",
            base_rate=0.51,
            weights={"reading_depth": -0.1, "breaking_news_mode": 0.3, "topic_affinity": 0.1},
        ),
        ArmDef(
            arm_id="balanced_mix",
            label="Balanced Mix",
            base_rate=0.55,
            weights={"reading_depth": 0.2, "breaking_news_mode": 0.05, "topic_affinity": 0.12},
        ),
        ArmDef(
            arm_id="deep_dive",
            label="Deep Dive",
            base_rate=0.49,
            weights={"reading_depth": 0.35, "breaking_news_mode": -0.2, "topic_affinity": 0.08},
        ),
    ),
)


CORE_WORLD_CONFIGS: tuple[WorldConfig, ...] = (
    RURAL_CLINIC_WORLD,
    MOVIEMATCH_WORLD,
    NEWSFEED_WORLD,
)
