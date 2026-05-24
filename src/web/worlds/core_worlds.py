"""Core narrative world configurations for Phase 2."""

from __future__ import annotations

from web.worlds.schema import ArmDef, FeatureDef, WorldConfig


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


SHOPSMART_WORLD = WorldConfig(
    world_id="shopsmart",
    title="ShopSmart",
    description="Optimize product recommendations for e-commerce conversion.",
    difficulty="easy",
    features=(
        FeatureDef(
            name="price_sensitivity", feature_type="numeric", numeric_min=0.0, numeric_max=1.0
        ),
        FeatureDef(
            name="loyalty_tier",
            feature_type="categorical",
            categories=("new", "bronze", "gold"),
        ),
        FeatureDef(name="mobile_user", feature_type="binary"),
    ),
    arms=(
        ArmDef(
            arm_id="discount_banner",
            label="Discount Banner",
            base_rate=0.55,
            weights={"price_sensitivity": 0.35, "loyalty_tier": -0.1, "mobile_user": 0.05},
        ),
        ArmDef(
            arm_id="premium_placement",
            label="Premium Placement",
            base_rate=0.48,
            weights={"price_sensitivity": -0.2, "loyalty_tier": 0.3, "mobile_user": -0.1},
        ),
        ArmDef(
            arm_id="social_proof",
            label="Social Proof",
            base_rate=0.52,
            weights={"price_sensitivity": 0.05, "loyalty_tier": 0.1, "mobile_user": 0.2},
        ),
    ),
)


RIDEPILOT_WORLD = WorldConfig(
    world_id="ridepilot",
    title="RidePilot",
    description="Dispatch ride-hailing requests to maximize rider satisfaction.",
    difficulty="medium",
    features=(
        FeatureDef(
            name="surge_multiplier", feature_type="numeric", numeric_min=1.0, numeric_max=3.0
        ),
        FeatureDef(name="trip_distance", feature_type="numeric", numeric_min=0.5, numeric_max=30.0),
        FeatureDef(
            name="time_of_day",
            feature_type="categorical",
            categories=("morning", "afternoon", "night"),
        ),
    ),
    arms=(
        ArmDef(
            arm_id="standard_dispatch",
            label="Standard Dispatch",
            base_rate=0.5,
            weights={"surge_multiplier": -0.1, "trip_distance": 0.05, "time_of_day": 0.05},
        ),
        ArmDef(
            arm_id="priority_routing",
            label="Priority Routing",
            base_rate=0.58,
            weights={"surge_multiplier": 0.15, "trip_distance": -0.1, "time_of_day": -0.05},
        ),
        ArmDef(
            arm_id="pool_match",
            label="Pool Match",
            base_rate=0.45,
            weights={"surge_multiplier": 0.25, "trip_distance": 0.15, "time_of_day": -0.1},
        ),
    ),
)


GAMEBOT_WORLD = WorldConfig(
    world_id="gamebot",
    title="GameBot",
    description="Adapt game difficulty to player skill for optimal engagement.",
    difficulty="medium",
    features=(
        FeatureDef(name="player_skill", feature_type="numeric", numeric_min=0.0, numeric_max=1.0),
        FeatureDef(
            name="session_count", feature_type="numeric", numeric_min=1.0, numeric_max=100.0
        ),
        FeatureDef(
            name="device_type",
            feature_type="categorical",
            categories=("mobile", "desktop", "console"),
        ),
    ),
    arms=(
        ArmDef(
            arm_id="easy_mode",
            label="Easy Mode",
            base_rate=0.55,
            weights={"player_skill": -0.25, "session_count": -0.05, "device_type": 0.1},
        ),
        ArmDef(
            arm_id="normal_mode",
            label="Normal Mode",
            base_rate=0.58,
            weights={"player_skill": 0.05, "session_count": 0.1, "device_type": 0.05},
        ),
        ArmDef(
            arm_id="hard_mode",
            label="Hard Mode",
            base_rate=0.48,
            weights={"player_skill": 0.3, "session_count": 0.15, "device_type": -0.1},
        ),
    ),
)


LABTRIAL_WORLD = WorldConfig(
    world_id="labtrial",
    title="LabTrial",
    description="Allocate clinical trial arms balancing efficacy and patient safety.",
    difficulty="hard",
    features=(
        FeatureDef(
            name="biomarker_level", feature_type="numeric", numeric_min=0.0, numeric_max=1.0
        ),
        FeatureDef(name="prior_response", feature_type="binary"),
        FeatureDef(
            name="risk_group",
            feature_type="categorical",
            categories=("low", "medium", "high"),
        ),
    ),
    arms=(
        ArmDef(
            arm_id="control",
            label="Control",
            base_rate=0.40,
            weights={"biomarker_level": -0.05, "prior_response": -0.1, "risk_group": 0.05},
        ),
        ArmDef(
            arm_id="low_dose",
            label="Low Dose",
            base_rate=0.52,
            weights={"biomarker_level": 0.15, "prior_response": 0.2, "risk_group": -0.1},
        ),
        ArmDef(
            arm_id="high_dose",
            label="High Dose",
            base_rate=0.45,
            weights={"biomarker_level": 0.25, "prior_response": 0.05, "risk_group": -0.3},
        ),
    ),
)


CORE_WORLD_CONFIGS: tuple[WorldConfig, ...] = (
    RURAL_CLINIC_WORLD,
    MOVIEMATCH_WORLD,
    NEWSFEED_WORLD,
    SHOPSMART_WORLD,
    RIDEPILOT_WORLD,
    GAMEBOT_WORLD,
    LABTRIAL_WORLD,
)
