# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic Data Generators
# MAGIC
# MAGIC Called by `01_setup.py`. Each function returns a pandas DataFrame.
# MAGIC All data is generated in-process — no external CSV dependencies.

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customers

# COMMAND ----------

def generate_customers(n=5000, seed=42):
    """Generate telecom customer profiles with realistic churn correlations (~26% churn rate)."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    from faker import Faker
    fake = Faker()
    Faker.seed(seed)

    contract_types = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    internet_services = ["Fiber optic", "DSL", "No"]

    records = []
    for i in range(n):
        customer_id = f"CUST-{i+1:05d}"
        name = fake.name()
        gender = rng.choice(["Male", "Female"])
        senior_citizen = int(rng.random() < 0.16)
        age = int(rng.integers(65, 85)) if senior_citizen else int(rng.integers(18, 65))
        partner = rng.choice(["Yes", "No"])
        dependents = rng.choice(["Yes", "No"], p=[0.3, 0.7])

        # Contract type influences churn
        contract_type = rng.choice(contract_types, p=[0.50, 0.25, 0.25])
        tenure_months = int(
            rng.integers(1, 12) if contract_type == "Month-to-month"
            else rng.integers(6, 48) if contract_type == "One year"
            else rng.integers(12, 72)
        )

        internet_service = rng.choice(internet_services, p=[0.44, 0.34, 0.22])
        phone_service = rng.choice(["Yes", "No"], p=[0.9, 0.1])

        # Optional services (only if internet)
        has_internet = internet_service != "No"
        streaming_tv = rng.choice(["Yes", "No"]) if has_internet else "No internet service"
        streaming_movies = rng.choice(["Yes", "No"]) if has_internet else "No internet service"
        online_security = rng.choice(["Yes", "No"], p=[0.35, 0.65]) if has_internet else "No internet service"
        online_backup = rng.choice(["Yes", "No"], p=[0.4, 0.6]) if has_internet else "No internet service"
        device_protection = rng.choice(["Yes", "No"], p=[0.4, 0.6]) if has_internet else "No internet service"
        tech_support = rng.choice(["Yes", "No"], p=[0.35, 0.65]) if has_internet else "No internet service"

        paperless_billing = rng.choice(["Yes", "No"], p=[0.6, 0.4])
        payment_method = rng.choice(payment_methods)

        # Monthly charges based on services
        base_charge = 20.0
        if internet_service == "Fiber optic":
            base_charge += 44.0
        elif internet_service == "DSL":
            base_charge += 25.0
        if phone_service == "Yes":
            base_charge += 10.0
        for svc in [streaming_tv, streaming_movies, online_security, online_backup, device_protection, tech_support]:
            if svc == "Yes":
                base_charge += rng.uniform(5, 12)
        monthly_charges = round(base_charge + rng.uniform(-5, 10), 2)
        total_charges = round(monthly_charges * tenure_months + rng.uniform(-50, 50), 2)
        total_charges = max(total_charges, monthly_charges)

        # Churn probability: higher for month-to-month, short tenure, high charges, no support services
        churn_score = 0.0
        if contract_type == "Month-to-month":
            churn_score += 0.25
        elif contract_type == "One year":
            churn_score += 0.05
        if tenure_months < 6:
            churn_score += 0.15
        elif tenure_months < 12:
            churn_score += 0.08
        if monthly_charges > 80:
            churn_score += 0.10
        elif monthly_charges > 60:
            churn_score += 0.05
        if online_security == "No" and tech_support == "No":
            churn_score += 0.08
        if payment_method == "Electronic check":
            churn_score += 0.06
        if internet_service == "Fiber optic":
            churn_score += 0.05
        churn_score = min(churn_score, 0.85)
        churn = "Yes" if rng.random() < churn_score else "No"

        records.append({
            "customer_id": customer_id,
            "name": name,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "age": age,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "phone_service": phone_service,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "paperless_billing": paperless_billing,
            "churn": churn,
        })

    return pd.DataFrame(records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Service Tickets

# COMMAND ----------

def generate_service_tickets(customers_df, n=15000, seed=42):
    """Generate service tickets correlated with churn behavior."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    categories = ["billing", "technical", "cancellation", "upgrade", "general"]
    priorities = ["low", "medium", "high", "critical"]
    statuses = ["open", "resolved", "escalated"]

    churner_ids = set(customers_df[customers_df["churn"] == "Yes"]["customer_id"])
    all_ids = list(customers_df["customer_id"])

    # Churners get ~3x more tickets
    weights = customers_df["churn"].map({"Yes": 3.0, "No": 1.0}).values
    weights = weights / weights.sum()

    ticket_descriptions = {
        "billing": [
            "Customer inquired about unexpected charge on their latest bill.",
            "Dispute over billing amount — customer claims overcharge for current cycle.",
            "Customer requesting breakdown of charges on monthly statement.",
            "Payment processing error reported — double charge on credit card.",
            "Customer asking about proration after mid-cycle plan change.",
            "Late fee dispute — customer says payment was submitted on time.",
            "Customer confused about taxes and surcharges on bill.",
            "Request to change billing date to align with pay schedule.",
        ],
        "technical": [
            "Intermittent internet connectivity — customer reports frequent drops.",
            "Slow download speeds not matching subscribed plan tier.",
            "Router not connecting after power outage — needs reset instructions.",
            "Customer unable to stream video — buffering issues during peak hours.",
            "Wi-Fi signal weak in parts of the home — requesting range extender.",
            "Email service not syncing on mobile device.",
            "DNS resolution failures reported across multiple devices.",
            "VoIP call quality degradation — choppy audio and latency.",
        ],
        "cancellation": [
            "Customer wants to cancel service — moving to a new provider.",
            "Requesting cancellation due to pricing — found cheaper alternative.",
            "Customer frustrated with service quality and wants to terminate.",
            "Cancellation request — customer relocating out of service area.",
            "Customer threatening to cancel unless given a better rate.",
            "Account cancellation requested — switching to competitor.",
        ],
        "upgrade": [
            "Customer interested in upgrading to a faster internet tier.",
            "Inquiry about adding streaming package to current plan.",
            "Customer wants to upgrade from DSL to fiber optic service.",
            "Request to add international calling to phone plan.",
            "Customer exploring premium channel add-on options.",
            "Inquiry about bundle discount for adding mobile service.",
        ],
        "general": [
            "Customer requesting update to account contact information.",
            "General inquiry about contract renewal terms.",
            "Customer asking about available loyalty rewards.",
            "Request for copy of service agreement.",
            "Customer inquiring about referral program benefits.",
            "Question about service availability at a new address.",
        ],
    }

    records = []
    base_date = datetime(2025, 1, 1)
    for i in range(n):
        cust_idx = rng.choice(len(all_ids), p=weights)
        customer_id = all_ids[cust_idx]
        is_churner = customer_id in churner_ids

        # Churners bias toward cancellation/billing
        if is_churner:
            cat = rng.choice(categories, p=[0.30, 0.20, 0.25, 0.05, 0.20])
        else:
            cat = rng.choice(categories, p=[0.20, 0.30, 0.05, 0.20, 0.25])

        priority = rng.choice(priorities, p=[0.30, 0.40, 0.20, 0.10])
        if cat == "cancellation":
            priority = rng.choice(priorities, p=[0.05, 0.20, 0.45, 0.30])

        status = rng.choice(statuses, p=[0.15, 0.65, 0.20])
        resolution_hours = round(float(rng.exponential(24)) + 0.5, 1) if status == "resolved" else None
        created_date = base_date + timedelta(days=int(rng.integers(0, 365)), hours=int(rng.integers(8, 20)))
        description = rng.choice(ticket_descriptions[cat])

        records.append({
            "ticket_id": f"TKT-{i+1:06d}",
            "customer_id": customer_id,
            "created_date": created_date,
            "category": cat,
            "priority": priority,
            "status": status,
            "resolution_time_hours": resolution_hours,
            "description": description,
        })

    return pd.DataFrame(records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call Transcripts

# COMMAND ----------

def _build_transcript_templates():
    """Return templates for call transcript generation."""

    positive_openings = [
        "Hi, I'm calling to ask about my account.",
        "Hello, I have a quick question about my service.",
        "Hi there, I'd like to get some information about my plan options.",
        "Good morning, I'm calling to check on something with my bill.",
    ]
    negative_openings = [
        "I need to talk to someone about my terrible service.",
        "I've been having problems for weeks and nobody has helped me.",
        "I'm really frustrated with the service I've been getting.",
        "Look, I'm seriously considering switching providers at this point.",
        "I've had it with these outages — I want to discuss my options.",
    ]
    billing_complaints = [
        "My bill went up by ${amount} this month and nobody told me about it. I've been a loyal customer for {tenure} months and this is how you treat me? I've seen ads from {competitor} offering much better rates.",
        "I was promised a rate of ${old_amount} per month but I'm being charged ${amount}. This has been going on for {months} months now and every time I call I get a different answer. {competitor} has a plan that's half the price.",
        "I don't understand why I'm paying ${amount} when I signed up for a completely different plan. My neighbor just switched to {competitor} and they're paying way less for the same service.",
    ]
    technical_issues = [
        "My internet has been dropping out every few hours for the past {days} days. I work from home and this is costing me productivity. I've already reset the router {times} times and nothing helps.",
        "The speeds I'm getting are nowhere near what I'm paying for. I ran a speed test and I'm only getting {speed}Mbps when I should be getting {expected}Mbps. This is unacceptable for what I'm paying.",
        "I can't stream anything without constant buffering. My family is frustrated — we're paying for premium service and getting budget quality. I've heard {competitor} has much more reliable fiber in our area.",
    ]
    cancellation_requests = [
        "I want to cancel my service effective immediately. I've already signed up with {competitor} because they offered me a better deal at ${amount} per month. Your service has been unreliable and overpriced.",
        "I'm calling to cancel. I've been a customer for {tenure} months and the service has only gotten worse while the price keeps going up. {competitor} is offering {offer} and honestly it's a no-brainer.",
        "Please cancel my account. I've tried working with your team multiple times but nothing changes. The outages, the billing errors, the long hold times — I'm done. {competitor} can't be any worse than this.",
    ]
    resolution_positive = [
        "Oh really? That does sound like a good offer. Let me think about it... okay, I'll stay if you can lock that rate in for {months} months. Thank you for working with me on this.",
        "I appreciate you looking into this. If you can fix the billing and give me a credit for the overcharges, I'm willing to give it another month. Thank you.",
        "That's actually a much better plan than what I have now. Okay, let's go ahead and make that change. I appreciate your help today.",
    ]
    resolution_negative = [
        "No, that's not good enough. I've already made up my mind. Please process the cancellation. I don't want any more calls from your retention team either.",
        "I've heard these promises before and nothing changes. Just cancel the account. I'll return the equipment this week.",
        "Look, I appreciate you trying, but I've already committed to {competitor}. The installation is scheduled for next week. Please just finalize the cancellation.",
    ]
    agent_responses = [
        "I completely understand your frustration, {customer_name}. Let me pull up your account and see what we can do.",
        "I'm sorry to hear about that experience, {customer_name}. Let me review your account right away.",
        "Thank you for your patience, {customer_name}. I can see the issue on your account and I want to make this right.",
        "I understand, {customer_name}. Let me check what options we have available for your account.",
    ]
    agent_offers = [
        "I can offer you a {discount}% discount on your current plan for the next 12 months, bringing your monthly bill down to ${new_amount}.",
        "We have a loyalty promotion I can apply — it would upgrade your speed tier at no additional cost for 6 months.",
        "I can waive the charges from last month and set up a credit of ${credit} on your account. I'll also escalate the technical issue to our priority support team.",
        "Let me see... I can move you to our {plan_name} plan which gives you more for ${new_amount} per month. That's actually ${savings} less than what you're paying now.",
    ]
    competitors = ["Spectrum", "Xfinity", "AT&T", "Verizon", "T-Mobile Home", "Google Fiber"]

    return {
        "positive_openings": positive_openings,
        "negative_openings": negative_openings,
        "billing_complaints": billing_complaints,
        "technical_issues": technical_issues,
        "cancellation_requests": cancellation_requests,
        "resolution_positive": resolution_positive,
        "resolution_negative": resolution_negative,
        "agent_responses": agent_responses,
        "agent_offers": agent_offers,
        "competitors": competitors,
    }


def generate_call_transcripts(customers_df, n=3000, seed=42):
    """Generate call transcripts with sentiment patterns correlated to churn."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    from faker import Faker
    fake = Faker()
    Faker.seed(seed)

    templates = _build_transcript_templates()
    churner_ids = set(customers_df[customers_df["churn"] == "Yes"]["customer_id"])
    cust_lookup = customers_df.set_index("customer_id")[["name", "tenure_months", "monthly_charges"]].to_dict("index")
    all_ids = list(customers_df["customer_id"])

    weights = customers_df["churn"].map({"Yes": 2.5, "No": 1.0}).values
    weights = weights / weights.sum()

    dispositions = ["resolved", "follow_up", "escalated", "transferred"]
    agent_names = [fake.name() for _ in range(20)]

    records = []
    base_date = datetime(2025, 1, 1)
    for i in range(n):
        cust_idx = rng.choice(len(all_ids), p=weights)
        customer_id = all_ids[cust_idx]
        is_churner = customer_id in churner_ids
        cust_info = cust_lookup[customer_id]
        customer_name = cust_info["name"].split()[0]
        tenure = cust_info["tenure_months"]
        charges = cust_info["monthly_charges"]
        competitor = rng.choice(templates["competitors"])

        # Build transcript
        parts = []
        if is_churner:
            opening = rng.choice(templates["negative_openings"])
            parts.append(f"Customer: {opening}")
        else:
            opening = rng.choice(templates["positive_openings"])
            parts.append(f"Customer: {opening}")

        agent_resp = rng.choice(templates["agent_responses"]).format(customer_name=customer_name)
        parts.append(f"Agent: {agent_resp}")

        # Main body
        if is_churner:
            body_type = rng.choice(["billing", "technical", "cancellation"], p=[0.3, 0.2, 0.5])
        else:
            body_type = rng.choice(["billing", "technical", "cancellation"], p=[0.4, 0.45, 0.15])

        fill = {
            "amount": int(charges + rng.integers(5, 25)),
            "old_amount": int(charges - rng.integers(10, 20)),
            "tenure": tenure,
            "months": int(rng.integers(2, 6)),
            "days": int(rng.integers(3, 14)),
            "times": int(rng.integers(2, 8)),
            "speed": int(rng.integers(15, 50)),
            "expected": int(rng.integers(100, 500)),
            "competitor": competitor,
            "offer": f"${int(rng.integers(30, 55))}/month with free installation",
        }
        if body_type == "billing":
            body = rng.choice(templates["billing_complaints"]).format(**fill)
        elif body_type == "technical":
            body = rng.choice(templates["technical_issues"]).format(**fill)
        else:
            body = rng.choice(templates["cancellation_requests"]).format(**fill)
        parts.append(f"Customer: {body}")

        # Agent offer
        offer_fill = {
            "discount": int(rng.integers(10, 30)),
            "new_amount": int(charges - rng.integers(10, 25)),
            "credit": int(rng.integers(25, 75)),
            "plan_name": rng.choice(["StreamMax Plus", "UltraFiber Pro", "ValueConnect Premium"]),
            "savings": int(rng.integers(10, 30)),
        }
        offer = rng.choice(templates["agent_offers"]).format(**offer_fill)
        parts.append(f"Agent: {offer}")

        # Resolution
        if is_churner and rng.random() < 0.6:
            resolution = rng.choice(templates["resolution_negative"]).format(competitor=competitor, months=int(rng.integers(3, 12)))
            disp = rng.choice(["escalated", "transferred"], p=[0.6, 0.4])
        else:
            resolution = rng.choice(templates["resolution_positive"]).format(months=int(rng.integers(6, 12)))
            disp = rng.choice(["resolved", "follow_up"], p=[0.7, 0.3])
        parts.append(f"Customer: {resolution}")
        parts.append(f"Agent: Thank you for calling, {customer_name}. Is there anything else I can help you with today?")
        parts.append(f"Customer: No, that's all. {'Goodbye.' if not is_churner else 'Just process what we discussed.'}")

        transcript_text = "\n".join(parts)
        call_date = base_date + timedelta(days=int(rng.integers(0, 365)), hours=int(rng.integers(8, 20)), minutes=int(rng.integers(0, 60)))
        call_duration = round(float(rng.uniform(4, 35)), 1)

        records.append({
            "transcript_id": f"CALL-{i+1:05d}",
            "customer_id": customer_id,
            "call_date": call_date,
            "agent_name": rng.choice(agent_names),
            "transcript_text": transcript_text,
            "call_duration_minutes": call_duration,
            "disposition": disp,
        })

    return pd.DataFrame(records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference Tables

# COMMAND ----------

def generate_plans():
    """Generate telecom plan catalog."""
    plans = [
        {"plan_id": "PLAN-001", "plan_name": "Basic Internet", "monthly_cost": 29.99, "data_limit_gb": 100, "minutes_included": 0, "text_included": 0, "contract_term": "Month-to-month", "features_description": "Basic DSL internet with 50Mbps download. Suitable for light browsing and email. Includes basic Wi-Fi router rental."},
        {"plan_id": "PLAN-002", "plan_name": "Standard Internet", "monthly_cost": 49.99, "data_limit_gb": 500, "minutes_included": 0, "text_included": 0, "contract_term": "Month-to-month", "features_description": "Standard fiber internet with 200Mbps download. Good for streaming and remote work. Includes dual-band Wi-Fi router."},
        {"plan_id": "PLAN-003", "plan_name": "Premium Internet", "monthly_cost": 79.99, "data_limit_gb": -1, "minutes_included": 0, "text_included": 0, "contract_term": "One year", "features_description": "Premium fiber internet with 1Gbps download. Unlimited data. Ideal for power users, gaming, and large households. Includes Wi-Fi 6E mesh router system."},
        {"plan_id": "PLAN-004", "plan_name": "UltraFiber Pro", "monthly_cost": 99.99, "data_limit_gb": -1, "minutes_included": 0, "text_included": 0, "contract_term": "One year", "features_description": "Ultra-premium 2Gbps symmetric fiber. Unlimited data with priority traffic during peak hours. Includes Wi-Fi 7 mesh system and dedicated support line."},
        {"plan_id": "PLAN-005", "plan_name": "Phone Basic", "monthly_cost": 15.99, "data_limit_gb": 0, "minutes_included": 500, "text_included": 500, "contract_term": "Month-to-month", "features_description": "Basic home phone with 500 minutes and 500 texts. Includes caller ID, voicemail, and call waiting."},
        {"plan_id": "PLAN-006", "plan_name": "Phone Unlimited", "monthly_cost": 24.99, "data_limit_gb": 0, "minutes_included": -1, "text_included": -1, "contract_term": "Month-to-month", "features_description": "Unlimited home phone with nationwide calling. Includes all premium features: caller ID, voicemail-to-email, 3-way calling, call forwarding."},
        {"plan_id": "PLAN-007", "plan_name": "StreamMax Bundle", "monthly_cost": 89.99, "data_limit_gb": -1, "minutes_included": -1, "text_included": -1, "contract_term": "One year", "features_description": "All-in-one bundle: 500Mbps fiber internet, unlimited phone, and premium streaming package (100+ channels). Best value for entertainment-focused households."},
        {"plan_id": "PLAN-008", "plan_name": "StreamMax Plus", "monthly_cost": 109.99, "data_limit_gb": -1, "minutes_included": -1, "text_included": -1, "contract_term": "One year", "features_description": "Premium bundle: 1Gbps fiber, unlimited phone, premium streaming with sports package, and cloud DVR (500 hours). Includes 4K streaming on all devices."},
        {"plan_id": "PLAN-009", "plan_name": "Business Starter", "monthly_cost": 59.99, "data_limit_gb": 500, "minutes_included": -1, "text_included": -1, "contract_term": "Two year", "features_description": "Small business internet and phone package. 300Mbps dedicated business line with static IP. Includes business-class router and priority support."},
        {"plan_id": "PLAN-010", "plan_name": "Business Pro", "monthly_cost": 129.99, "data_limit_gb": -1, "minutes_included": -1, "text_included": -1, "contract_term": "Two year", "features_description": "Professional business package: 1Gbps symmetric fiber with SLA guarantees. Unlimited phone with auto-attendant. Includes managed firewall and 24/7 dedicated support."},
        {"plan_id": "PLAN-011", "plan_name": "ValueConnect", "monthly_cost": 39.99, "data_limit_gb": 250, "minutes_included": 250, "text_included": 250, "contract_term": "Month-to-month", "features_description": "Budget-friendly internet and phone combo. 100Mbps internet with basic phone service. No contract required. Great for cost-conscious customers."},
        {"plan_id": "PLAN-012", "plan_name": "ValueConnect Premium", "monthly_cost": 59.99, "data_limit_gb": 500, "minutes_included": -1, "text_included": -1, "contract_term": "Month-to-month", "features_description": "Mid-tier combo: 200Mbps internet with unlimited phone. No contract lock-in. Includes standard streaming package with 50+ channels."},
        {"plan_id": "PLAN-013", "plan_name": "Senior Essentials", "monthly_cost": 34.99, "data_limit_gb": 200, "minutes_included": -1, "text_included": -1, "contract_term": "Month-to-month", "features_description": "Designed for seniors: 100Mbps internet, unlimited phone, simplified remote control for TV. Includes in-home setup assistance and dedicated senior support line."},
        {"plan_id": "PLAN-014", "plan_name": "Student Connect", "monthly_cost": 29.99, "data_limit_gb": 300, "minutes_included": 0, "text_included": 0, "contract_term": "Month-to-month", "features_description": "Student-discounted internet: 200Mbps with no contract. Valid with student ID. Includes free antivirus and parental controls."},
        {"plan_id": "PLAN-015", "plan_name": "Gamer Elite", "monthly_cost": 89.99, "data_limit_gb": -1, "minutes_included": 0, "text_included": 0, "contract_term": "One year", "features_description": "Optimized for gaming: 1Gbps low-latency fiber with QoS prioritization for gaming traffic. Includes gaming-optimized router with port forwarding and DDoS protection."},
    ]
    return pd.DataFrame(plans)

# COMMAND ----------

def generate_policies():
    """Generate company policy documents."""
    policies = [
        {
            "policy_id": "POL-001",
            "policy_name": "Standard Cancellation Policy",
            "category": "cancellation",
            "effective_date": datetime(2024, 6, 1),
            "policy_text": "Customers may cancel their service at any time by contacting our support team. Month-to-month customers can cancel with no early termination fee. Customers on annual or two-year contracts who cancel early will be charged an early termination fee of $10 per remaining month on the contract. Equipment must be returned within 14 days of cancellation to avoid equipment charges. Final bills will be prorated to the cancellation date. Any promotional credits will be forfeited upon cancellation. A 30-day notice period is recommended but not required. Cancellation requests are processed within 2 business days."
        },
        {
            "policy_id": "POL-002",
            "policy_name": "Retention Offer Guidelines",
            "category": "retention",
            "effective_date": datetime(2024, 9, 1),
            "policy_text": "Retention specialists may offer the following incentives to at-risk customers: (1) Monthly discount of 10-25% for 6-12 months based on customer tenure and lifetime value. Customers with 24+ months tenure qualify for up to 25% discount. (2) Free speed tier upgrade for 6 months. (3) One-time account credit of $25-$75 for service issues. (4) Free premium add-on (streaming, security, etc.) for 3-6 months. (5) Contract buyout consideration for customers switching from a competitor. Maximum combined offer value should not exceed $500 per customer per year. All offers must be documented in the CRM system. Escalate to supervisor for offers exceeding standard guidelines."
        },
        {
            "policy_id": "POL-003",
            "policy_name": "Billing Dispute Resolution",
            "category": "billing",
            "effective_date": datetime(2024, 3, 1),
            "policy_text": "All billing disputes must be investigated within 5 business days. Agents can issue immediate credits up to $50 without supervisor approval. Credits between $50-$150 require team lead approval. Credits over $150 require manager approval. Recurring billing errors must be escalated to the billing systems team. Customers disputing charges older than 90 days should be directed to the formal dispute process. Proration adjustments for plan changes are automatic. Late fees can be waived once per 12-month period as a courtesy. Payment plan arrangements are available for balances over $200."
        },
        {
            "policy_id": "POL-004",
            "policy_name": "Service Level Agreement",
            "category": "billing",
            "effective_date": datetime(2024, 1, 1),
            "policy_text": "Residential customers are guaranteed 99.5% network uptime measured monthly. Business customers on Pro plans are guaranteed 99.9% uptime. If uptime falls below the guaranteed level, customers are eligible for a service credit equal to 1/30th of their monthly charge per day of downtime beyond the SLA threshold. Outages must be reported within 48 hours. Scheduled maintenance windows (typically Tuesday and Thursday 2-6 AM) are excluded from uptime calculations. Speed guarantees are 80% of advertised download speeds measured at the router. Customers experiencing persistent speed issues may request a technician visit at no charge."
        },
        {
            "policy_id": "POL-005",
            "policy_name": "Plan Upgrade and Downgrade Policy",
            "category": "upgrade",
            "effective_date": datetime(2024, 4, 1),
            "policy_text": "Customers can upgrade their plan at any time with changes taking effect within 24 hours. Upgrades during a contract do not extend the contract term. Downgrades are allowed once per 6-month period for contracted customers. Month-to-month customers can downgrade at any time. When upgrading, the new rate is prorated from the change date. When downgrading, the current rate applies through the end of the billing cycle. Bundled services can be modified individually. Equipment upgrades associated with plan changes may require a one-time installation fee of $49.99, waived for customers with 12+ months tenure."
        },
        {
            "policy_id": "POL-006",
            "policy_name": "Customer Data Privacy Policy",
            "category": "general",
            "effective_date": datetime(2024, 1, 1),
            "policy_text": "Customer personal information is collected and used solely for service provisioning, billing, and support. Data is not sold to third parties. Customers may request a copy of their data or deletion of their account data per applicable privacy regulations. Call recordings are retained for 90 days for quality assurance. Service usage data is anonymized and may be used for network planning. Customers can opt out of marketing communications at any time. Agent access to customer data is logged and audited. Two-factor authentication is available for account security."
        },
        {
            "policy_id": "POL-007",
            "policy_name": "Equipment Return Policy",
            "category": "cancellation",
            "effective_date": datetime(2024, 6, 1),
            "policy_text": "All leased equipment (routers, modems, set-top boxes) must be returned within 14 days of service cancellation. Equipment can be returned at any retail location, via prepaid shipping label, or through scheduled pickup. Unreturned equipment will result in charges: router/modem $150, set-top box $200, mesh extender $100. Damaged equipment beyond normal wear is charged at 75% of replacement cost. Equipment purchased outright by the customer does not need to be returned. Customers receive a confirmation email when returned equipment is processed."
        },
        {
            "policy_id": "POL-008",
            "policy_name": "Loyalty Rewards Program",
            "category": "retention",
            "effective_date": datetime(2024, 7, 1),
            "policy_text": "Customers earn loyalty points based on tenure and monthly spend: 1 point per $1 spent per month, with a 1.5x multiplier after 12 months and 2x after 24 months. Points can be redeemed for: account credits (1000 pts = $10), free premium add-ons (2000 pts = 1 month), equipment upgrades (5000 pts), or partner gift cards (variable). Loyalty tier levels: Silver (0-12 months), Gold (12-24 months), Platinum (24+ months). Platinum members receive priority support queue, annual equipment refresh, and exclusive promotional offers. Points expire after 18 months of account inactivity."
        },
    ]
    return pd.DataFrame(policies)

# COMMAND ----------

def generate_product_knowledge():
    """Generate knowledge base articles for RAG retrieval."""
    articles = [
        {"article_id": "KB-001", "title": "Troubleshooting Slow Internet Speeds", "category": "troubleshooting", "last_updated": datetime(2025, 1, 15),
         "content": "If you are experiencing slow internet speeds, follow these steps to diagnose and resolve the issue. First, run a speed test at speedtest.net while connected directly to your router via ethernet cable. This eliminates Wi-Fi as a variable. If wired speeds match your plan, the issue is Wi-Fi related. Try repositioning your router to a central location, away from walls, metal objects, and other electronics. Ensure your router firmware is up to date by checking the admin panel at 192.168.1.1. If wired speeds are also slow, power cycle your modem and router by unplugging both for 30 seconds, then plugging in the modem first and waiting 2 minutes before plugging in the router. Check for bandwidth-heavy applications running in the background such as cloud backups, system updates, or streaming on other devices. During peak hours (7-11 PM), some congestion is normal on shared network segments. If speeds consistently fall below 80% of your subscribed tier, contact support for a line quality test. Our technicians can check signal levels, noise ratios, and provisioning remotely. If a physical issue is detected, we will schedule a technician visit at no charge."},
        {"article_id": "KB-002", "title": "Understanding Your Monthly Bill", "category": "faq", "last_updated": datetime(2025, 2, 1),
         "content": "Your monthly bill consists of several components. The base service charge reflects your subscribed plan rate. If you are on a promotional rate, the bill will show both the standard rate and the promotional discount as a separate line item. Taxes and regulatory fees are government-mandated charges that vary by location and typically add 8-15% to your base rate. These include federal Universal Service Fund fees, state telecom taxes, and local franchise fees. Equipment charges cover leased hardware such as routers and set-top boxes. If you purchased your own compatible equipment, this line will not appear. One-time charges may include installation fees, technician visits, or activation charges. Pay-per-view or on-demand purchases appear as separate line items. Late payment fees of $10 are applied to balances unpaid after 30 days. If you see unexpected charges, check the 'Account Activity' section in your online portal for a detailed breakdown. You can also set up autopay to receive a $5/month discount and avoid late fees. Bill disputes should be filed within 90 days of the charge date."},
        {"article_id": "KB-003", "title": "Wi-Fi Optimization Guide", "category": "troubleshooting", "last_updated": datetime(2025, 1, 20),
         "content": "Optimizing your home Wi-Fi network can dramatically improve your internet experience. Start by understanding the two frequency bands available on modern routers. The 2.4GHz band offers better range but slower speeds and is more susceptible to interference from microwaves, baby monitors, and neighbors' networks. The 5GHz band is faster but has shorter range. Wi-Fi 6 and 6E routers add the 6GHz band for even faster, less congested connections. For best results, connect devices that need speed (streaming, gaming, video calls) to 5GHz or 6GHz, and IoT devices (smart home, sensors) to 2.4GHz. Place your router in a central, elevated location. Each wall or floor between your device and the router reduces signal strength by approximately 25-50%. If you have dead zones, consider our mesh extender add-on which creates a seamless network throughout your home. Use the channel selection feature in your router settings to avoid congestion — tools like Wi-Fi Analyzer can show which channels are least crowded. Keep your network secure with WPA3 encryption and a strong password to prevent unauthorized users from consuming your bandwidth."},
        {"article_id": "KB-004", "title": "Comparing Our Internet Plans", "category": "plan_comparison", "last_updated": datetime(2025, 2, 10),
         "content": "Choosing the right internet plan depends on your household size, usage patterns, and budget. Our Basic Internet plan (50Mbps, $29.99/mo) is suitable for 1-2 people who primarily browse the web, check email, and do light streaming. It supports one HD stream comfortably. Our Standard Internet plan (200Mbps, $49.99/mo) handles 2-4 people with moderate usage including multiple simultaneous HD streams, video conferencing, and online gaming. This is our most popular residential plan. The Premium Internet plan (1Gbps, $79.99/mo) is designed for power users and large households of 4+ people. It handles multiple 4K streams, large file downloads, competitive gaming, and smart home devices simultaneously without slowdown. Our UltraFiber Pro plan (2Gbps, $99.99/mo) is our flagship offering with symmetric upload and download speeds, priority traffic routing during peak hours, and a Wi-Fi 7 mesh system included. It is ideal for content creators, remote workers handling large files, and tech-savvy households. All fiber plans include unlimited data. DSL plans (Basic) have a 100GB soft cap with no overage charges but may experience reduced speeds during peak hours after the cap is reached."},
        {"article_id": "KB-005", "title": "Setting Up Your New Router", "category": "how_to", "last_updated": datetime(2025, 1, 5),
         "content": "Setting up your new router is straightforward and should take about 15 minutes. Unbox the router and connect the power adapter. Use the included ethernet cable to connect the WAN/Internet port on your router to the LAN port on your modem. If you have a combined modem-router unit, skip this step. Wait 2-3 minutes for the router to boot and establish a connection (the internet LED should turn solid green). Connect to the default Wi-Fi network — the network name (SSID) and password are printed on the label on the bottom of the router. Open a web browser and navigate to the setup page (typically 192.168.1.1 or the address on the label). The setup wizard will guide you through creating a custom network name and password, selecting your time zone, and enabling automatic firmware updates. We strongly recommend changing the default admin password for security. For mesh extender setup, place the extender halfway between your router and the dead zone, then press the WPS button on both devices within 2 minutes. The extender LED will turn solid when paired. You can also use our mobile app for guided setup with step-by-step instructions and automated optimization."},
        {"article_id": "KB-006", "title": "Streaming Quality and Bandwidth Requirements", "category": "feature_guide", "last_updated": datetime(2025, 1, 25),
         "content": "Streaming video quality depends directly on your available bandwidth. Standard definition (SD/480p) requires approximately 3 Mbps. High definition (HD/1080p) needs 5-8 Mbps per stream. Ultra high definition (4K/2160p) requires 25 Mbps per stream, and 8K content needs approximately 100 Mbps. Remember these are per-stream requirements — a household with three people streaming simultaneously in HD needs at least 24 Mbps of available bandwidth. Music streaming services use 0.3-1.5 Mbps depending on quality settings. Video conferencing platforms like Zoom require 3-4 Mbps for HD video calls and 5-8 Mbps for group calls. Online gaming itself uses relatively little bandwidth (1-3 Mbps) but is highly sensitive to latency and packet loss. Our Premium and UltraFiber plans include QoS (Quality of Service) settings that you can configure to prioritize streaming or gaming traffic. Access QoS settings through your router admin panel under Advanced > Traffic Management. Our streaming package add-on ($15/mo) includes access to 100+ live channels and on-demand content library. Premium streaming ($25/mo) adds sports channels, 4K content, and cloud DVR with 500 hours of storage."},
        {"article_id": "KB-007", "title": "Home Phone Features and Setup", "category": "feature_guide", "last_updated": datetime(2024, 12, 15),
         "content": "Our VoIP home phone service includes a comprehensive set of features. Standard features available on all plans include caller ID with name display, call waiting, voicemail with up to 30 saved messages, and 911 emergency calling. Our Unlimited Phone plan adds three-way calling, call forwarding to any number, voicemail-to-email transcription, anonymous call rejection, and do-not-disturb scheduling. To set up your phone service, connect a standard phone to the Phone port on your modem/router using a regular phone cable. Dial *99 to access the automated setup menu where you can record your voicemail greeting and configure call forwarding rules. Online portal configuration is available at myaccount.telecom.com where you can manage all call features, view call history, and set up simultaneous ring (rings your home phone and cell phone at the same time). International calling is available as an add-on: $10/mo for unlimited calling to Canada and Mexico, $20/mo for the International 60 package covering 60 countries. Per-minute rates apply for countries not included in your package."},
        {"article_id": "KB-008", "title": "Online Security and Device Protection", "category": "feature_guide", "last_updated": datetime(2025, 2, 5),
         "content": "Our Online Security add-on ($9.99/mo) provides comprehensive protection for your connected devices. It includes real-time antivirus and anti-malware scanning powered by industry-leading threat intelligence. The service protects up to 10 devices across Windows, Mac, iOS, and Android platforms. Features include safe browsing alerts that warn you before visiting malicious websites, a password manager for securely storing credentials, and a VPN service for encrypted browsing on public Wi-Fi networks. Parental controls allow you to set content filters by age group, schedule internet access times, and monitor browsing activity for each device. Our Device Protection plan ($7.99/mo) covers repair or replacement of your connected devices including computers, tablets, and smartphones for hardware failure, accidental damage, and power surge damage. Coverage includes up to $500 per claim with a $29 deductible. File up to 3 claims per year. Both services can be managed through our Security Center app or the online portal. We recommend enabling automatic scans scheduled during low-usage hours and keeping the software agent updated for the best protection."},
        {"article_id": "KB-009", "title": "Upgrading Your Service Plan", "category": "how_to", "last_updated": datetime(2025, 1, 30),
         "content": "Upgrading your plan is simple and can be done through several channels. Through the online portal: log into myaccount.telecom.com, navigate to My Plan, and click Upgrade. You will see available upgrade options with pricing. Select your desired plan and confirm — changes typically take effect within 24 hours. Through our mobile app: open the app, tap Services, then Change Plan, and follow the prompts. By phone: call our sales team and they can process the upgrade immediately. Upgrades during a contract do not extend your contract term. Your bill will be prorated for the remainder of the current billing cycle at the new rate. If your upgrade requires new equipment (for example, moving from DSL to fiber), a technician installation visit will be scheduled. Installation is free for customers with 12+ months tenure and $49.99 for newer customers. The technician will install the new equipment, verify speeds, and ensure your network is optimized. After an upgrade, we recommend running a speed test to confirm you are receiving the new plan speeds. Allow 24-48 hours for network provisioning to fully complete. If speeds do not match after 48 hours, contact support for a line check."},
        {"article_id": "KB-010", "title": "Cancellation and Account Closure Process", "category": "faq", "last_updated": datetime(2025, 2, 1),
         "content": "If you are considering cancelling your service, we would like the opportunity to address your concerns first. Our retention team can often resolve issues or offer competitive alternatives. To cancel, call our support line — cancellations cannot be processed online for security purposes. A retention specialist will review your account, discuss any outstanding concerns, and present available offers. If you choose to proceed with cancellation, here is what to expect: your service will remain active through the end of the current billing cycle. Your final bill will include any remaining balance plus applicable early termination fees for contracted plans ($10 per remaining contract month). You have 14 days from the cancellation date to return leased equipment to any retail location or request a prepaid return shipping label. A $150-$200 equipment charge applies for unreturned hardware. Any remaining loyalty points will expire upon account closure. Account data is retained for 60 days in case you wish to reactivate, after which it is permanently deleted per our privacy policy. You can port your phone number to another provider before or during the cancellation process — coordinate with your new provider to initiate the port."},
        {"article_id": "KB-011", "title": "Business Internet Solutions Overview", "category": "plan_comparison", "last_updated": datetime(2025, 1, 10),
         "content": "Our business internet solutions are designed for reliability, performance, and scalability. The Business Starter plan ($59.99/mo) provides 300Mbps dedicated bandwidth with a static IP address, suitable for small offices of 1-10 employees handling email, web browsing, and cloud applications. The Business Pro plan ($129.99/mo) delivers 1Gbps symmetric fiber with SLA-backed 99.9% uptime guarantee, multiple static IPs, managed firewall, and 24/7 dedicated business support with guaranteed 4-hour response time. For larger organizations, our Enterprise plan (custom pricing) offers dedicated fiber connections up to 10Gbps, redundant paths for failover, MPLS networking for multi-site connectivity, and a dedicated account manager. All business plans include free professional installation, business-class router with advanced QoS and VLAN support, and priority repair scheduling. We offer month-to-month flexibility with a 10% premium or cost savings with 2-year agreements. Business customers can add VoIP phone systems with auto-attendant, call routing, and conference bridge capabilities. Our SD-WAN add-on provides intelligent traffic routing across multiple internet connections for maximum reliability."},
        {"article_id": "KB-012", "title": "Troubleshooting No Internet Connection", "category": "troubleshooting", "last_updated": datetime(2025, 2, 8),
         "content": "If you have completely lost internet connectivity, follow this systematic troubleshooting process. Step 1: Check for service outages in your area by visiting status.telecom.com on your mobile device or calling our automated status line. If an outage is confirmed, no further action is needed — service will be restored automatically. Step 2: Verify all cable connections. Ensure the coaxial or fiber cable from the wall is securely connected to your modem, and the ethernet cable from modem to router is plugged in at both ends. Look for any bent or damaged cables. Step 3: Check the modem lights. The Power light should be solid green. The Online/Internet light should be solid green or blue — if it is flashing or red, the modem cannot reach our network. The LAN light should be active if a device is connected via ethernet. Step 4: Power cycle your equipment. Unplug both your modem and router. Wait 30 seconds. Plug in the modem first and wait until all status lights stabilize (approximately 2 minutes). Then plug in the router and wait another 2 minutes. Step 5: Try connecting a device directly to the modem via ethernet, bypassing the router entirely. If this works, the issue is with your router configuration. Step 6: If none of these steps resolve the issue, contact our technical support team. Have your account number ready and note which modem lights are active — this helps our technicians diagnose the problem quickly."},
        {"article_id": "KB-013", "title": "Parental Controls and Content Filtering", "category": "how_to", "last_updated": datetime(2025, 1, 12),
         "content": "Our parental controls system gives you comprehensive tools to manage your family's internet experience. Access controls through the router admin panel at 192.168.1.1 or via our mobile app. Device-level controls let you set different rules for each connected device. Create profiles for each family member and assign their devices to the appropriate profile. Age-based content filters automatically block inappropriate websites and content categories. Choose from preset levels: Child (ages 5-8, most restrictive), Pre-teen (ages 9-12), Teen (ages 13-17), or create custom filter rules. Schedule internet access by setting allowed hours for each profile — for example, no internet after 9 PM on school nights. Bedtime mode gradually reduces available services, first blocking games and social media, then all internet access. The Activity Reports section shows browsing history, top visited sites, and blocked content attempts for each profile. Reports can be emailed to you weekly. Our Online Security add-on enhances parental controls with YouTube filtering (restricts to YouTube Kids content), search engine safe mode enforcement, app blocking by category, and real-time alerts when blocked content is accessed. All settings sync across our router, mesh extenders, and the mobile app."},
        {"article_id": "KB-014", "title": "Moving Your Service to a New Address", "category": "how_to", "last_updated": datetime(2025, 1, 18),
         "content": "Moving to a new address does not mean you need to cancel and restart service. Our Move Service option transfers your account, plan, and promotional rates to your new location. Start the process at least 2 weeks before your move date by calling our support team or visiting the Move section in your online portal. First, we will verify service availability at your new address — our coverage includes most metropolitan and suburban areas. If your current plan is available at the new address, we will schedule installation for your preferred date. If your current technology is not available (for example, you have fiber but only DSL is available at the new address), we will discuss alternative plan options and any price adjustments. Installation at the new address is free for customers with 12+ months tenure. You can choose to keep your current equipment or receive new hardware at the new location. Your phone number can typically be ported to the new address if it is within the same rate center — we will confirm during the move request. Service at your old address will be disconnected on your move date. Return any equipment staying at the old address within 14 days using our prepaid shipping label."},
        {"article_id": "KB-015", "title": "Understanding Data Caps and Usage", "category": "faq", "last_updated": datetime(2025, 2, 3),
         "content": "Data caps apply to some of our plans and understanding how they work helps you manage your usage effectively. DSL and Basic plans include a soft data cap — this means you will not be charged for exceeding the limit, but speeds may be reduced during peak hours once you pass the threshold. Fiber plans (Standard and above) include unlimited data with no caps or throttling. You can monitor your current data usage in real time through the online portal or mobile app under Usage > Data. We also send email alerts at 75% and 90% of your cap. Average monthly data usage by household: light usage (email, browsing) averages 50-100GB; moderate usage (streaming, social media) averages 200-400GB; heavy usage (4K streaming, gaming, cloud backups) averages 500GB-1TB+. Common high-bandwidth activities and their approximate data consumption: streaming 1 hour of HD video uses about 3GB, 4K video uses about 7GB; downloading a modern video game averages 50-100GB; a one-hour Zoom call uses about 1.5GB. If you consistently exceed your cap, consider upgrading to a plan with a higher allowance or unlimited data. Our Usage Advisor tool analyzes your patterns and recommends the most cost-effective plan for your household."},
        {"article_id": "KB-016", "title": "Network Security Best Practices", "category": "feature_guide", "last_updated": datetime(2025, 1, 22),
         "content": "Protecting your home network is essential in today's connected world. Start with your router password — change it from the default immediately. Use a strong, unique password with at least 12 characters including letters, numbers, and symbols. Enable WPA3 encryption on your Wi-Fi network (or WPA2 if your devices do not support WPA3). Never use WEP or open networks. Create a separate guest network for visitors and IoT devices — this isolates your primary devices from potentially vulnerable smart home gadgets. Keep your router firmware updated; enable automatic updates if available. Disable WPS (Wi-Fi Protected Setup) as it has known vulnerabilities. Disable remote management unless you specifically need it. Consider using our DNS-level content filtering which blocks known malicious domains before they can reach your devices. Our Online Security add-on provides additional protection with real-time threat monitoring, automatic malware blocking, and encrypted DNS. Review connected devices regularly through your router admin panel — if you see unfamiliar devices, change your Wi-Fi password immediately. For critical tasks like online banking, ensure the website shows HTTPS (padlock icon) in the browser. Be cautious with public Wi-Fi; use our included VPN service when connecting to networks outside your home."},
        {"article_id": "KB-017", "title": "Loyalty Rewards Program Details", "category": "faq", "last_updated": datetime(2025, 2, 12),
         "content": "Our Loyalty Rewards Program automatically enrolls all customers and rewards your continued business with points redeemable for valuable perks. You earn 1 point per $1 spent on your monthly bill. After 12 months of service, your earning rate increases to 1.5x, and after 24 months, you earn at 2x rate. Points can be redeemed in several ways: Account Credits — 1,000 points equals a $10 credit on your next bill. Premium Add-ons — 2,000 points gets you one month of any premium add-on (streaming, security, etc.). Equipment Upgrades — 5,000 points toward a router upgrade or mesh extender. Partner Gift Cards — redeem points for gift cards from popular retailers and restaurants. Loyalty tiers provide additional benefits: Silver (0-12 months) gets standard support and basic rewards; Gold (12-24 months) adds priority support queue and exclusive quarterly promotions; Platinum (24+ months) includes dedicated support line, annual equipment refresh eligibility, and early access to new products and services. Check your points balance and tier status in the Rewards section of your online portal or mobile app. Points expire after 18 months of account inactivity (no billing activity). Promotional bonus point events are offered periodically — watch for notifications in your account."},
        {"article_id": "KB-018", "title": "Troubleshooting Router Connectivity Issues", "category": "troubleshooting", "last_updated": datetime(2025, 1, 28),
         "content": "Router connectivity problems are among the most common issues and are usually resolved with these steps. If devices connect to Wi-Fi but show 'No Internet': this typically means the router is working but cannot reach our network. Check the internet light on your router — if it is red or off, the issue is between your router and our network. Power cycle the modem first, then the router. If specific devices cannot connect while others work fine, the issue is device-specific. On the affected device, forget the Wi-Fi network and reconnect with the password. On Windows, run the network troubleshooter (Settings > Network > Troubleshoot). On Mac, renew the DHCP lease (System Preferences > Network > Advanced > TCP/IP > Renew DHCP Lease). On mobile devices, toggle airplane mode on and off. If devices connect to 2.4GHz but not 5GHz (or vice versa), check that both bands are enabled in your router settings and that the device supports the band. If Wi-Fi disconnects intermittently, check for interference sources: microwave ovens, cordless phones, baby monitors, and neighboring Wi-Fi networks on the same channel. Use the Wi-Fi Analyzer feature in our app to find the least congested channel and switch to it. If problems persist after trying these steps, a factory reset of the router may be necessary — press and hold the reset button for 10 seconds, then reconfigure using the setup wizard."},
        {"article_id": "KB-019", "title": "Bundle Discounts and Package Options", "category": "plan_comparison", "last_updated": datetime(2025, 2, 6),
         "content": "Bundling multiple services together offers significant savings compared to purchasing each service individually. Our StreamMax Bundle ($89.99/mo) combines 500Mbps internet, unlimited phone, and our premium streaming package — separately these would cost $134.97/mo, saving you $44.98 monthly. The StreamMax Plus Bundle ($109.99/mo) upgrades to 1Gbps internet and adds the sports package with cloud DVR — individual cost would be $164.97/mo, a savings of $54.98 per month. Custom bundles are also available: add any phone plan to an internet plan for 20% off the phone plan rate. Add streaming to any internet plan for $15/mo (standard) or $25/mo (premium with sports and 4K). Add Online Security to any bundle for $7.99/mo instead of the standalone $9.99/mo. Add Device Protection to any bundle for $5.99/mo instead of $7.99/mo standalone. Multi-service discount tiers: 2 services bundled saves 15%, 3 services saves 20%, 4+ services saves 25% off the combined individual rates. All bundle discounts apply for the duration of your service — they are not promotional and will not expire. Contract-term bundles (1 or 2 year) receive an additional 5% discount. You can modify individual components of your bundle without losing the bundle discount, as long as you maintain at least 2 services."},
        {"article_id": "KB-020", "title": "Smart Home Integration Guide", "category": "feature_guide", "last_updated": datetime(2025, 1, 8),
         "content": "Our internet service fully supports smart home ecosystems and we offer tools to optimize your connected home experience. Most smart home devices (smart speakers, lights, thermostats, cameras) use the 2.4GHz Wi-Fi band, so ensure this band is enabled and has good coverage throughout your home. Our mesh extender system is particularly beneficial for smart homes as it provides consistent coverage to all corners of your home where devices may be placed. For homes with 10+ smart devices, we recommend our Standard plan (200Mbps) or higher to ensure sufficient bandwidth, even though individual IoT devices use minimal data. In your router settings, we recommend creating a dedicated IoT network — this is a separate SSID for your smart devices that keeps them isolated from your computers and phones for security purposes. Our router supports UPnP (Universal Plug and Play) for automatic device discovery and IFTTT integration for automation routines. For voice assistant integration, our service is compatible with Amazon Alexa, Google Assistant, and Apple HomeKit. You can use voice commands to check your internet status, run speed tests, and control guest network access. Our mobile app includes a Connected Devices section that shows all devices on your network with real-time bandwidth usage, helping you identify if any device is consuming excessive resources."},
        # Additional articles for variety
        {"article_id": "KB-021", "title": "Email Service Configuration", "category": "how_to", "last_updated": datetime(2025, 1, 14),
         "content": "Your internet plan includes up to 5 email accounts with our domain. To set up your email, visit mail.telecom.com and click Create Account. Choose your preferred address format (firstname.lastname@telecom.com is recommended). Each mailbox includes 10GB of storage. Configure email on your devices using these settings: Incoming server (IMAP): imap.telecom.com, port 993, SSL enabled. Outgoing server (SMTP): smtp.telecom.com, port 587, TLS enabled. Use your full email address as the username. Our webmail interface at mail.telecom.com provides access from any browser with features including folder organization, contact management, calendar integration, and 25MB attachment limit per email. Spam filtering is enabled by default and can be customized in Settings > Mail Filters. You can create rules to automatically sort incoming mail into folders, forward specific messages, or block senders. If you are switching from another email provider, our Migration Assistant tool can import your old emails, contacts, and calendar entries. Access it from Settings > Import. Anti-virus scanning automatically checks all incoming attachments. For business customers, we offer enhanced email with 50GB storage, custom domain support, and shared calendars for $5/user/month."},
        {"article_id": "KB-022", "title": "Understanding Service Outages", "category": "faq", "last_updated": datetime(2025, 2, 9),
         "content": "Service outages can occur for various reasons and understanding them helps set expectations for resolution. Planned maintenance is scheduled during low-usage windows (typically Tuesday and Thursday, 2-6 AM local time) and is announced at least 48 hours in advance via email and the status page. These rarely exceed 30 minutes. Unplanned outages fall into several categories: network equipment failure (typically resolved within 2-4 hours as redundant systems activate and engineers replace hardware), fiber cuts from construction or weather (4-24 hours depending on location and extent), power grid outages affecting our facilities (resolved when power is restored, our facilities have generator backup for 8+ hours), and major weather events (variable timeline, prioritized by impact). During an outage, check status.telecom.com for real-time updates including estimated restoration time, affected areas, and root cause. You can also call our automated status line or check our social media accounts for updates. Service credits are automatically applied for outages exceeding your plan's SLA threshold. If you experience frequent localized outages, please report them — this helps our network team identify infrastructure that needs proactive maintenance or upgrade."},
        {"article_id": "KB-023", "title": "Fiber vs DSL: Which Technology Is Right for You", "category": "plan_comparison", "last_updated": datetime(2025, 1, 20),
         "content": "Understanding the difference between our fiber and DSL technologies helps you choose the right service. DSL (Digital Subscriber Line) uses existing copper telephone lines to deliver internet. Our DSL plans offer speeds up to 50Mbps download and 10Mbps upload. DSL speeds decrease with distance from our central office — homes within 1 mile get optimal performance. DSL is available in most areas where we provide phone service and requires no new infrastructure installation. It is our most affordable option and sufficient for basic to moderate internet usage. Fiber optic internet uses thin glass strands that transmit data as pulses of light. Fiber offers dramatically faster speeds (200Mbps to 2Gbps), symmetric upload and download speeds, lower latency (important for gaming and video calls), and is not affected by distance or electromagnetic interference. Fiber requires installation of new cabling to your home, which our technicians handle during a scheduled appointment. The installation process takes 2-4 hours and is free for customers committing to 12+ months of service. If fiber is available at your address, we strongly recommend it for future-proofing — as households add more connected devices and streaming services, bandwidth demands will only increase. Check availability at your address by entering your zip code on our website or calling our sales team."},
        {"article_id": "KB-024", "title": "VPN Service and Privacy Features", "category": "feature_guide", "last_updated": datetime(2025, 2, 4),
         "content": "Our VPN (Virtual Private Network) service, included with the Online Security add-on, encrypts your internet traffic and protects your privacy when browsing. The VPN creates a secure tunnel between your device and our servers, preventing anyone on the same network from seeing your online activity. This is especially important when using public Wi-Fi at coffee shops, airports, or hotels. Our VPN network includes servers in 20+ countries, allowing you to connect from different geographic locations. The service supports all major platforms: Windows, macOS, iOS, Android, and Linux. Install the app from your device's app store or download from our website. Connect with one click — the app automatically selects the fastest server, or you can choose a specific location. The VPN supports split tunneling, which lets you route some traffic through the VPN while other traffic (like streaming) goes directly through your regular connection. This optimizes speed for services that work best with a direct connection. Business customers can use our VPN for secure remote access to company resources. Our business VPN supports site-to-site connections and integrates with common enterprise authentication systems. Speeds through the VPN are typically 85-95% of your regular connection speed due to the encryption overhead."},
        {"article_id": "KB-025", "title": "Troubleshooting VoIP Phone Issues", "category": "troubleshooting", "last_updated": datetime(2025, 1, 25),
         "content": "VoIP (Voice over IP) phone quality depends on your internet connection. If you experience choppy audio, echo, or dropped calls, follow these steps. First, check your internet speed — VoIP requires at least 500Kbps per active call. Run a speed test while the issue occurs to check for bandwidth problems. If your internet speed is normal, the issue may be network congestion within your home. Enable QoS (Quality of Service) on your router and set voice traffic to highest priority. Access this in your router settings under Advanced > QoS or Traffic Management. If you hear echo on calls, it is usually caused by the handset volume being too high — reduce the speaker volume by 20-30%. Echo can also occur with speakerphones in rooms with hard surfaces that reflect sound. For dropped calls, check that your modem and router firmware are up to date. Outdated firmware can cause SIP registration failures that interrupt calls. If the phone has no dial tone, verify the phone cable connection to the Phone port on your modem. Try a different phone or cable to rule out hardware issues. Power cycle the modem and wait 5 minutes for the phone service to re-register. If you are using a cordless phone system, interference from other wireless devices can affect call quality. Try changing the cordless phone channel or switching to a DECT 6.0 model which operates on a dedicated frequency band."},
        # Fill remaining articles with varied topics
        {"article_id": "KB-026", "title": "Autopay and Payment Options", "category": "how_to", "last_updated": datetime(2025, 2, 1),
         "content": "We offer several convenient payment methods to manage your account. Autopay is the easiest option — link a credit card, debit card, or bank account to automatically pay your bill each month. Autopay customers receive a $5/month discount and never worry about late fees. Enroll through your online portal under Billing > Payment Methods > Set Up Autopay. One-time payments can be made online, through the mobile app, by phone (automated system available 24/7), or by mail. We accept Visa, Mastercard, American Express, Discover, and electronic bank transfers. Pay by text: text PAY to our service number and follow the prompts. In-person payments can be made at any of our retail locations. Payment is applied to your account immediately for electronic methods; mailed payments may take 3-5 business days to process. If you are experiencing financial hardship, we offer payment arrangement plans that split your balance into manageable installments over 3-6 months with no additional fees. Contact our billing team to discuss options. Budget billing is also available — we calculate your average monthly charge over the past 12 months and bill that flat amount each month, with an annual true-up to reconcile any difference. This helps avoid seasonal fluctuations in your bill."},
        {"article_id": "KB-027", "title": "Speed Test Interpretation Guide", "category": "troubleshooting", "last_updated": datetime(2025, 1, 30),
         "content": "Understanding your speed test results helps you determine if your service is performing correctly. Download speed measures how fast data flows from the internet to your device — this affects streaming, web browsing, and file downloads. Upload speed measures how fast data flows from your device to the internet — this affects video calls, file sharing, cloud backups, and live streaming. Latency (ping) measures the time in milliseconds for data to make a round trip to a server. Low latency (under 20ms) is excellent for gaming and video calls. Latency under 50ms is good for most activities. Jitter measures the variation in latency — consistent low jitter is important for real-time applications. For accurate results: test with a wired ethernet connection to your router, close all other applications and stop other devices from using the network, run the test at different times of day (morning, afternoon, evening), and run the test 3 times and average the results. Expected speeds: you should see at least 80% of your subscribed plan speed on a wired connection. Wi-Fi speeds are typically 50-70% of wired speeds due to overhead and environmental factors. If your results are consistently below expectations, note the time, connection type, and speeds achieved, then contact support with this information for a line quality investigation."},
        {"article_id": "KB-028", "title": "Seasonal Weather Impact on Service", "category": "faq", "last_updated": datetime(2025, 1, 5),
         "content": "Weather can affect internet and phone service depending on your connection type. DSL customers may experience slower speeds or intermittent connectivity during heavy rain, as water can infiltrate older copper line connections and degrade the signal. If you consistently experience weather-related issues, request a line quality inspection — our technicians can identify and repair vulnerable connection points. Fiber optic cables are not affected by moisture or electromagnetic interference, making fiber service inherently more weather-resistant. However, extreme events like ice storms or falling trees can physically damage aerial fiber lines. Underground fiber is protected from most weather events. Severe thunderstorms can cause power surges that damage networking equipment. We strongly recommend using a surge protector for your modem and router. For customers in areas prone to power outages, consider a UPS (Uninterruptible Power Supply) to keep your internet running during short outages — a basic UPS can power a modem and router for 1-2 hours. During major weather events, our network operations center proactively monitors for damage and pre-positions repair crews. Check status.telecom.com for real-time outage maps and restoration estimates. We automatically apply service credits for weather-related outages exceeding 24 hours."},
        {"article_id": "KB-029", "title": "Accessible Services and Disability Support", "category": "faq", "last_updated": datetime(2025, 1, 8),
         "content": "We are committed to making our services accessible to all customers. Our Senior Essentials plan includes simplified remote controls with larger buttons and clear labeling, in-home setup assistance at no extra charge, and a dedicated support line with extended call times and patient, trained representatives. For customers who are deaf or hard of hearing, our phone service supports TTY/TDD devices and is compatible with captioned telephone services. Video relay service (VRS) works seamlessly over our internet connection with recommended plans of Standard Internet or higher. Our website and mobile app meet WCAG 2.1 AA accessibility standards, including screen reader compatibility, keyboard navigation, high contrast mode, and adjustable font sizes. Closed captioning is available on all streaming content through our entertainment packages. For customers with visual impairments, we offer bills and correspondence in large print, Braille, or audio format upon request. Our IVR (phone menu) system supports voice navigation as an alternative to keypad input. If you need assistance with equipment setup or have specific accessibility requirements, our dedicated accessibility support team can be reached to arrange personalized installation and configuration. We also participate in the FCC Lifeline program, providing discounted service to qualifying low-income households."},
        {"article_id": "KB-030", "title": "Cloud DVR and On-Demand Content Guide", "category": "feature_guide", "last_updated": datetime(2025, 2, 7),
         "content": "Our Cloud DVR service lets you record and watch your favorite shows on your schedule. Included with StreamMax Plus bundles and available as an add-on ($10/mo) for other streaming plans, Cloud DVR stores recordings in the cloud so you can access them from any device — TV, tablet, phone, or computer. Standard Cloud DVR includes 200 hours of HD storage. Premium upgrade (included with StreamMax Plus) provides 500 hours. Record up to 6 programs simultaneously so you never miss conflicting shows. Recordings are kept for 12 months. Use the program guide to schedule one-time or series recordings. Series recordings automatically capture new episodes and can be set to keep all episodes or only a specified number. Fast forward through commercials in recorded content using the 15-second skip or standard fast forward controls. On-demand content is available through our streaming interface with over 50,000 titles including movies, TV series, documentaries, and kids programming. New releases are typically available 30-90 days after theatrical release. Our recommendation engine learns your preferences and suggests content you are likely to enjoy. Create up to 6 viewer profiles per household, each with its own watchlist and recommendations. Parental controls can restrict on-demand content by rating (G, PG, PG-13, R, etc.) for each profile."},
    ]

    # Generate remaining 20 articles programmatically for variety
    additional_topics = [
        ("KB-031", "Gaming Network Optimization", "feature_guide", "Optimize your network for gaming by enabling QoS settings to prioritize game traffic. Use a wired ethernet connection for competitive gaming to minimize latency. Our Premium and UltraFiber plans include gaming optimization profiles that automatically prioritize traffic from popular game servers. Port forwarding can be configured for specific games through the router admin panel — navigate to Advanced > Port Forwarding and add entries for your game's required ports. Enable UPnP for games that dynamically assign ports. For the best experience, ensure your NAT type is set to Open (Type 1) or Moderate (Type 2). Our gaming-optimized Gamer Elite plan routes traffic through dedicated low-latency paths to major game server locations."),
        ("KB-032", "Remote Work Setup Recommendations", "feature_guide", "Working from home requires reliable internet and proper network configuration. We recommend at minimum our Standard plan (200Mbps) for remote workers, or Premium (1Gbps) if you regularly transfer large files or participate in multiple video calls simultaneously. Position your workspace near the router or use a wired ethernet connection for the most stable experience. If wired is not possible, our mesh extender system ensures strong Wi-Fi throughout your home. Enable QoS to prioritize video conferencing traffic. For VPN connections to your employer, ensure your upload speed is sufficient — most business VPNs require at least 10Mbps upload for smooth operation. Consider our Online Security add-on for additional protection when accessing company resources from home."),
        ("KB-033", "Mesh Wi-Fi Extender Setup and Placement", "troubleshooting", "Our mesh Wi-Fi extenders create a seamless wireless network that covers your entire home. Each extender covers approximately 1,500 square feet. For optimal placement, position the first extender halfway between your router and the area with weak signal. The extender should show at least 2 out of 3 signal strength bars from the router's location. Avoid placing extenders near large metal objects, aquariums, or thick concrete walls. Each additional extender should be within range of at least one other mesh node. Our system supports up to 6 extenders for large homes. Setup is automatic — press the sync button on both the router and extender within 2 minutes. The extender creates a seamless network using the same SSID, so devices automatically connect to the strongest node as you move through your home. Check mesh health in our mobile app under Network > Mesh Status."),
        ("KB-034", "International Calling Plans and Rates", "plan_comparison", "Stay connected with family and friends abroad with our international calling options. The North America plan ($10/mo) includes unlimited calls to Canada and Mexico. The International 60 package ($20/mo) covers unlimited calling to 60 countries including the UK, Germany, France, India, China, Japan, Australia, and most of Western Europe and Asia. For countries not included in your package, per-minute rates apply and vary by destination — check our rate card online at telecom.com/international-rates. International texting is $0.25 per message or unlimited with any international calling package. Calls are made from your regular home phone number with no special dialing codes needed beyond the country code."),
        ("KB-035", "Account Security and Two-Factor Authentication", "how_to", "Protect your account with our security features. Two-factor authentication (2FA) adds an extra layer of security by requiring a verification code in addition to your password when logging in. Enable 2FA in your online portal under Settings > Security > Two-Factor Authentication. Choose to receive codes via text message or authenticator app (Google Authenticator, Authy, etc.). We also support security questions as a backup verification method. Set a unique PIN for phone support interactions — this prevents unauthorized persons from making changes to your account over the phone. Review your account activity regularly under Settings > Login History to spot any unauthorized access. If you suspect your account has been compromised, change your password immediately and contact our support team. We will review recent account activity and reverse any unauthorized changes."),
        ("KB-036", "TV Streaming Package Channel Lineup", "plan_comparison", "Our streaming packages bring you the content you love without a cable box. The Standard Streaming Package ($15/mo add-on) includes 50+ channels covering news (CNN, MSNBC, Fox News), entertainment (TNT, TBS, USA, FX), lifestyle (HGTV, Food Network, Discovery), and kids programming (Cartoon Network, Nickelodeon, Disney Channel). The Premium Streaming Package ($25/mo) adds everything in Standard plus premium channels (HBO, Showtime, Starz), the complete sports package (ESPN networks, regional sports, NFL Network), 4K content on supported channels, and Cloud DVR with 500 hours of storage. All streaming packages include on-demand libraries with thousands of movies and TV shows."),
        ("KB-037", "Troubleshooting Email Delivery Issues", "troubleshooting", "If you are not receiving emails or your sent messages are bouncing, try these solutions. For missing incoming mail, check your Spam/Junk folder first — legitimate emails sometimes get filtered. Add trusted senders to your contacts or whitelist to prevent future filtering. Verify your mailbox is not full (10GB limit) by checking storage usage in webmail settings. For bounced outgoing messages, verify the recipient's email address is correct. Our SMTP server limits outgoing messages to 500 per day to prevent spam abuse — if you need higher limits for business use, contact support. If you see a specific bounce code, look it up: 550 means the recipient address does not exist, 552 means the recipient's mailbox is full, and 554 typically means your message was flagged as spam by the recipient's server."),
        ("KB-038", "Equipment Warranty and Replacement", "faq", "All leased equipment is covered by our comprehensive warranty for the duration of your service. If your router, modem, or other leased equipment malfunctions, we will replace it at no charge. Contact support to arrange a replacement — we can ship a new unit (arrives in 2-3 business days) or you can exchange at any retail location same-day. If you purchased your own equipment and it fails, check if it is still under the manufacturer's warranty. We maintain a list of approved third-party equipment at telecom.com/approved-devices. Equipment that fails due to power surges is covered under our Device Protection plan. Without Device Protection, equipment damaged by acts of nature or customer misuse may incur replacement charges."),
        ("KB-039", "Network Maintenance and Upgrade Schedule", "faq", "We continuously invest in our network infrastructure to improve speed, reliability, and coverage. Regular maintenance windows are Tuesday and Thursday from 2-6 AM local time. Most maintenance activities cause no service interruption, but when they do, disruptions typically last less than 30 minutes. Major network upgrades are communicated via email at least 2 weeks in advance. Our current infrastructure expansion projects include extending fiber coverage to additional neighborhoods, upgrading core network equipment to support 10Gbps backbone connections, and deploying DOCSIS 4.0 technology for improved DSL performance. Customers in upgrade areas will be notified when faster speeds become available at their address. We publish a quarterly network performance report on our website showing uptime statistics, average speeds by area, and planned improvements."),
        ("KB-040", "Referring Friends and Family", "faq", "Our referral program rewards you for recommending our service to friends and family. For each successful referral, you receive a $50 account credit and your referred friend gets $25 off their first month. There is no limit to the number of referrals you can make. To refer someone, log into your online portal and navigate to Rewards > Refer a Friend. Enter your friend's name and email address, and we will send them a personalized offer link. The referral is tracked automatically — when your friend signs up using the link and completes their first month of service, both credits are applied. You can also share your unique referral code directly. Referral credits are applied within one billing cycle of the new customer's activation. Combine referral credits with our loyalty points for maximum savings. Top referrers each quarter are entered into a drawing for a free year of service."),
    ]

    for aid, title, category, content in additional_topics:
        articles.append({
            "article_id": aid,
            "title": title,
            "category": category,
            "last_updated": datetime(2025, 1, int(np.random.default_rng(42).integers(1, 28))),
            "content": content,
        })

    # Pad to 50 articles with short variations
    more_topics = [
        ("KB-041", "Scheduling a Technician Visit", "how_to", "If your issue requires on-site assistance, our technicians are available Monday through Saturday from 8 AM to 7 PM. Schedule a visit through your online portal, mobile app, or by calling support. Same-day appointments may be available in your area. You will receive a 2-hour arrival window and a text notification 30 minutes before the technician arrives. Standard technician visits are free for issues related to our network or leased equipment. Visits for customer-owned equipment or inside wiring may incur a $75 service charge. Ensure someone 18 or older is home during the appointment. The technician will test line quality, signal levels, and equipment performance, and replace any faulty leased hardware on the spot."),
        ("KB-042", "Understanding Latency and Ping", "feature_guide", "Latency, measured in milliseconds (ms), is the time it takes for data to travel from your device to a server and back. Low latency is crucial for real-time applications like online gaming, video conferencing, and VoIP calls. Fiber connections typically offer 5-15ms latency to nearby servers, while DSL averages 25-50ms. Factors affecting latency include distance to the server, network congestion, number of network hops, and your connection type. To reduce latency: use a wired connection instead of Wi-Fi, close bandwidth-heavy applications, enable QoS to prioritize time-sensitive traffic, and choose servers geographically close to you. Our gaming and business plans include optimized routing that reduces latency to major service endpoints."),
        ("KB-043", "Mobile App Features Overview", "how_to", "Our mobile app (available for iOS and Android) puts account management at your fingertips. View and pay your bill, monitor data usage in real time, run speed tests, manage Wi-Fi settings, view connected devices, set up parental controls, troubleshoot common issues with guided diagnostics, chat with support, schedule technician visits, and manage your streaming content and DVR recordings. The app also provides push notifications for outages in your area, bill due dates, and data usage alerts. Download from the App Store or Google Play by searching for our company name. Log in with your online portal credentials."),
        ("KB-044", "Protecting Against Phishing and Scams", "feature_guide", "We will never ask for your full password, Social Security number, or payment information via email or text message. If you receive a suspicious message claiming to be from us, do not click any links — instead, log into your account directly through our official website or app to verify any claims. Common scam indicators include urgent language, threats of service disconnection, requests to call non-official phone numbers, and offers that seem too good to be true. Report suspicious messages by forwarding them to abuse@telecom.com. Our Online Security add-on includes phishing protection that warns you before visiting known scam websites and blocks malicious email attachments automatically."),
        ("KB-045", "Temporary Service Suspension", "how_to", "If you will be away for an extended period (vacation, seasonal home, etc.), you can temporarily suspend your service for 1-6 months. During suspension, you pay a reduced rate of $9.99/month to maintain your account, phone number, and email address. Your equipment remains installed and service can be reactivated within 24 hours when you return. Request suspension through the online portal or by calling support at least 3 days before the desired start date. Your contract term is paused during suspension and resumes when service is reactivated. Promotional rates are preserved through one suspension per year. This option is not available for accounts with past-due balances."),
        ("KB-046", "Understanding Your Contract Terms", "faq", "Month-to-month plans have no commitment and can be changed or cancelled at any time. One-year and two-year contracts offer lower monthly rates in exchange for a service commitment. Early termination fees apply if you cancel before the contract end date: $10 per remaining month. Contract terms begin on your service activation date and auto-renew as month-to-month at the same rate when the term expires. You can switch to a different contract at any time — upgrading to a longer term locks in a lower rate but resets the commitment period. Downgrading to month-to-month from a contract incurs the early termination fee. Review your contract details and renewal date in your online portal under Account > Contract Details."),
        ("KB-047", "Multi-Room Streaming Setup", "how_to", "Watch your streaming content on every TV in your home without additional cable boxes. Our streaming service works on most smart TVs (Samsung, LG, Sony, Vizio 2018+), streaming devices (Roku, Apple TV, Amazon Fire TV, Chromecast), gaming consoles (PlayStation, Xbox), and mobile devices. Download our streaming app from your device's app store and log in with your account credentials. You can stream on up to 3 devices simultaneously with Standard Streaming and 5 devices with Premium Streaming. For TVs without smart capabilities, a streaming stick (available from us for $29.99 or use your own) plugs into any HDMI port. Each TV can have its own viewer profile with personalized recommendations and watch history."),
        ("KB-048", "Troubleshooting Set-Top Box Issues", "troubleshooting", "If your set-top box is not working correctly, start with a power cycle: unplug the box, wait 30 seconds, and plug it back in. Wait 3-5 minutes for it to fully reboot and download the program guide. If the screen shows 'No Signal', verify the HDMI cable connection and that your TV is set to the correct input. If channels are missing from your lineup, the box may need a signal refresh — go to your online portal and click Refresh Box under Support > Equipment. For frozen screens or sluggish menus, check if a software update is pending (Settings > System > Software Update) and install it. Audio/video sync issues are usually resolved by switching the audio output format in Settings from surround to stereo and back."),
        ("KB-049", "Green Energy and Sustainability Initiatives", "faq", "We are committed to reducing our environmental footprint. Our data centers and network facilities are transitioning to 100% renewable energy sources, currently at 75% renewable. We participate in equipment recycling programs — when you return old hardware, it is refurbished for reuse or responsibly recycled. Our paperless billing option reduces paper waste and earns you a $2/month eco-discount. The energy-efficient equipment we provide meets Energy Star certification standards, consuming up to 40% less power than previous generation hardware. We plant one tree for every new fiber installation completed. Track our sustainability progress on our website's Corporate Responsibility page."),
        ("KB-050", "Emergency Services and 911 Calling", "faq", "Our VoIP phone service includes enhanced 911 (E911) capability that transmits your registered service address to emergency dispatchers. It is critical that you keep your service address up to date in your account settings — if you move, update your address before placing emergency calls from the new location. Unlike traditional landlines, VoIP 911 requires internet connectivity and power to function. During power outages, your phone service will not work unless you have battery backup (UPS). We recommend maintaining a charged mobile phone as a backup for emergency calls. If you dial 911 from our phone service, the call is routed to your local Public Safety Answering Point (PSAP) along with your name and registered address. For customers with medical alert systems, ensure the system is configured to work with VoIP — most modern systems are compatible, but older units designed for analog lines may require an adapter."),
    ]

    for aid, title, category, content in more_topics:
        articles.append({
            "article_id": aid,
            "title": title,
            "category": category,
            "last_updated": datetime(2025, 1, int(np.random.default_rng(hash(aid) % 2**31).integers(1, 28))),
            "content": content,
        })

    return pd.DataFrame(articles)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Churn Labels

# COMMAND ----------

def generate_churn_labels(customers_df, seed=42):
    """Extract churn labels into separate table with train/val/test split."""
    rng = np.random.default_rng(seed)

    labels = customers_df[["customer_id", "churn"]].copy()
    labels["label_date"] = datetime(2025, 6, 1)

    # 70/15/15 split
    n = len(labels)
    splits = rng.choice(["train", "val", "test"], size=n, p=[0.70, 0.15, 0.15])
    labels["split"] = splits

    return labels
