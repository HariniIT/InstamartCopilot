
import os
import json

# --------------------------------------------------
# 1. OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR = "data/raw/refund_policies"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. REFUND POLICY DOCUMENTS
# --------------------------------------------------

policies = [

    {
        "policy_id": "REF001",
        "title": "General Refund Policy",
        "category": "General",

        "content": """
INSTAMART GENERAL REFUND POLICY

1. Customers can request a refund for eligible orders.
2. Refund requests must include a valid order ID.
3. Refund eligibility depends on order status and issue type.
4. Duplicate refund requests may be rejected.
5. Refund approval is subject to verification.
6. Customers will receive a refund status update.
7. If a request cannot be verified, it must be reviewed by support.

These are demonstration policies for the Instamart Copilot project.
"""
    },

    {
        "policy_id": "REF002",
        "title": "Damaged Product Refund Policy",
        "category": "Product Quality",

        "content": """
INSTAMART DAMAGED PRODUCT REFUND POLICY

1. Customers should report damaged products with their order ID.
2. Customers should provide photographs of the damaged product.
3. The support team must verify the submitted evidence.
4. Eligible damaged products can receive a refund.
5. Refund approval depends on verification.
6. If the product is unsafe to consume, customers should not consume it.
7. Unverified damage claims must be escalated to human support.

These are demonstration policies for the Instamart Copilot project.
"""
    },

    {
        "policy_id": "REF003",
        "title": "Missing Item Refund Policy",
        "category": "Missing Item",

        "content": """
INSTAMART MISSING ITEM REFUND POLICY

1. Customers can report missing items from their order.
2. Customers must provide the order ID.
3. The support team should compare the ordered items with delivery records.
4. Verified missing items may qualify for a partial refund.
5. The refund amount depends on the missing item's price.
6. Multiple missing item complaints require manual verification.
7. Unresolved cases must be escalated to human support.

These are demonstration policies for the Instamart Copilot project.
"""
    },

    {
        "policy_id": "REF004",
        "title": "Payment Failure Refund Policy",
        "category": "Payment",

        "content": """
INSTAMART PAYMENT FAILURE REFUND POLICY

1. Customers should provide the order ID or transaction reference.
2. Payment status must be checked before approving a refund.
3. If money was deducted but the order was not created,
   the transaction must be verified.
4. Duplicate payment claims require transaction verification.
5. Refund approval depends on the verified payment status.
6. If payment status is unclear, the case must be escalated.
7. The AI assistant must never claim that a refund was completed
   without confirmation from the payment system.

These are demonstration policies for the Instamart Copilot project.
"""
    },

    {
        "policy_id": "REF005",
        "title": "Refund Escalation Policy",
        "category": "Escalation",

        "content": """
INSTAMART REFUND ESCALATION POLICY

1. Unverified refund requests must be sent to human support.
2. High-priority customer complaints should be reviewed quickly.
3. Repeated payment failures require manual investigation.
4. Customers should receive an explanation of escalation status.
5. The AI assistant must not invent refund approval decisions.
6. The system should maintain a record of customer complaints.
7. Human support has final authority over unresolved refund cases.

These are demonstration policies for the Instamart Copilot project.
"""
    }

]

# --------------------------------------------------
# 3. SAVE INDIVIDUAL POLICY DOCUMENTS
# --------------------------------------------------

for policy in policies:

    filename = policy["policy_id"] + ".txt"

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(policy["title"])
        file.write("\n\n")
        file.write(policy["content"])

    print("Created:", filepath)

# --------------------------------------------------
# 4. SAVE POLICY METADATA
# --------------------------------------------------

metadata_path = os.path.join(
    OUTPUT_DIR,
    "policy_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        policies,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nMetadata saved:", metadata_path)

# --------------------------------------------------
# 5. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("REFUND POLICY KNOWLEDGE BASE CREATED SUCCESSFULLY!")
print("=" * 60)

print("Total policy documents:", len(policies))
