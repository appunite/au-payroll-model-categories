-- Fetch training data for expense tag prediction
-- This query extracts invoices with expense tags, unnesting the tag array
-- to create one row per tag for training

WITH main AS (
    SELECT
        i."entityId",
        i."ownerId",
        i."issueDate",
        i."netPrice",
        i."grossPrice",
        i.currency,
        i."titleNormalized" as invoice_title,

        regexp_replace(regexp_replace(
                CASE WHEN bt."beneficiaryTin" IS NOT NULL THEN
                    bt."beneficiaryTin"
                WHEN length(t."documentData") > 0 THEN
                    json_entity.value ->> 'mentionText'
                ELSE
                    NULL
                END,
                '^\s*PL',
                '',
                'gi'),
            '[\s\-–—]',
            '',
            'g') AS tin,
        i."expenseTags"
    FROM
        invoices i
    LEFT JOIN "files" t ON t.id = i."fileId"
    LEFT JOIN "bankTransfers" bt ON bt."id" = i."bankTransferId"
    LEFT JOIN LATERAL json_array_elements(
        CASE WHEN length(t."documentData") > 0 THEN
            CAST(convert_from(t."documentData",
                    'UTF8') AS json) -> 'document' -> 'entities'
        ELSE
            '[]'::json
        END) AS json_entity (value) ON json_entity.value ->> 'type' = 'supplier_tax_id'
WHERE
    i."issueDate" > '2024-01-01'
    AND i."expenseCategory" <> 'others:contractors'
    AND t."documentData" IS NOT NULL
    AND i.status <> 'voided'
    AND i."titleNormalized" is NOT NULL
    AND t."template" = 'budgetInvoice'
    AND (i."expenseTags" <> '{}'
        AND i."expenseTags" IS NOT NULL)
ORDER BY
    i."titleNormalized",
    i."expenseCategory"
)
SELECT
    main.*,
    tag
FROM
    main
    LEFT JOIN LATERAL unnest(main."expenseTags") AS tag ON TRUE
WHERE
    tag IN(
        'visual-panda',
        'referral-fee',
        'legal-advice',
        'esop',
        'dashbit-jose-valim',
        'benefit-training',
        'benefit-psychologist',
        'benefit-outing',
        'benefit-multisport',
        'benefit-medical-care',
        'benefit-insurance',
        'benefit-english',
        'benefit-computer-formula',
        'benefit-books-formula',
        'benefit-apartments',
        'accounting',
        'BHP'
    )
