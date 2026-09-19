"""
MedFlow patient arrival generator.

Generates patients during a simulated 24-hour hospital day.
Arrival rates vary by time of day:
- Higher demand during morning and evening
- Lower demand overnight

This module only generates patients. The AI triage model will later
predict acuity from the patient features.
"""

import random
from typing import Any


SIMULATION_MINUTES = 24 * 60

# Average arrivals per hour for each time period.
# These values can be tuned later using real hospital data.
ARRIVAL_RATE_PER_HOUR = {
    "night": 3,       # 00:00–06:00
    "morning": 8,     # 06:00–12:00
    "afternoon": 6,   # 12:00–18:00
    "evening": 10,    # 18:00–24:00
}

POSSIBLE_RESOURCES = [
    "bed",
    "doctor",
    "nurse",
    "icu_bed",
    "operating_room",
]

COMPLAINTS = [
    "chest pain",
    "broken arm",
    "high fever",
    "minor cut",
    "difficulty breathing",
    "stomach pain",
    "headache",
    "car accident injury",
]


def get_arrival_rate_per_hour(time_minute: float) -> float:
    """Return the expected patient arrival rate for a given minute."""

    hour = int(time_minute // 60) % 24

    if 0 <= hour < 6:
        return ARRIVAL_RATE_PER_HOUR["night"]
    if 6 <= hour < 12:
        return ARRIVAL_RATE_PER_HOUR["morning"]
    if 12 <= hour < 18:
        return ARRIVAL_RATE_PER_HOUR["afternoon"]

    return ARRIVAL_RATE_PER_HOUR["evening"]


def create_one_patient(patient_id: int, arrival_time: float) -> dict[str, Any]:
    """Create one patient record.

    Acuity is intentionally left as None. The AI triage module will
    predict it later from the patient's features.
    """

    complaint = random.choice(COMPLAINTS)

    patient = {
        "id": patient_id,
        "arrival_time": round(arrival_time, 1),
        "complaint": complaint,
        "age": random.randint(1, 95),
        "heart_rate": random.randint(55, 160),
        "blood_pressure": random.randint(85, 190),
        "oxygen_level": random.randint(85, 100),
        "temperature": round(random.uniform(36.0, 40.5), 1),
        "respiratory_rate": random.randint(10, 35),
        "needs": random.sample(
            POSSIBLE_RESOURCES,
            random.randint(1, 3),
        ),
        "predicted_acuity": None,
        "predicted_los": None,
    }

    return patient


def generate_patients_for_one_day(
    seed: int | None = 42,
) -> list[dict[str, Any]]:
    """Generate patients using a time-varying Poisson arrival process.

    The thinning method is used:
    1. Generate candidate arrivals using the maximum arrival rate.
    2. Accept or reject each candidate according to the arrival rate
       for that time of day.

    Returns:
        Patients sorted by arrival time.
    """

    if seed is not None:
        random.seed(seed)

    patients: list[dict[str, Any]] = []

    current_time = 0.0
    patient_id = 1

    maximum_rate_per_hour = max(ARRIVAL_RATE_PER_HOUR.values())
    maximum_rate_per_minute = maximum_rate_per_hour / 60.0

    while current_time < SIMULATION_MINUTES:
        # Candidate gap from the maximum-rate Poisson process.
        gap = random.expovariate(maximum_rate_per_minute)
        current_time += gap

        if current_time >= SIMULATION_MINUTES:
            break

        actual_rate = get_arrival_rate_per_hour(current_time) / 60.0

        # Thinning: accept candidate with probability actual/max rate.
        acceptance_probability = actual_rate / maximum_rate_per_minute

        if random.random() <= acceptance_probability:
            patient = create_one_patient(patient_id, current_time)
            patients.append(patient)
            patient_id += 1

    return patients
